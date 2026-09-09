import traceback
from typing import Dict, List, Optional

from trinity.buffer.buffer import BufferWriter, get_buffer_reader, get_buffer_writer
from trinity.buffer.operators.experience_operator import ExperienceOperator
from trinity.buffer.storage.queue import is_database_url, is_json_file
from trinity.common.action_event_contract import validate_on_policy_experiences
from trinity.common.config import (
    AlgorithmConfig,
    BufferConfig,
    Config,
    ExperiencePipelineConfig,
    StorageConfig,
)
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.utils.log import get_logger
from trinity.utils.plugin_loader import load_plugins


DIAGNOSTIC_EXPERIENCE_SCHEMA_VERSION = "agemem.bench_experience_audit.v1"


def get_input_buffers(
    pipeline_config: ExperiencePipelineConfig, buffer_config: BufferConfig
) -> Dict:
    """Get input buffers for the experience pipeline."""
    input_buffers = {}
    for input_name, input_config in pipeline_config.inputs.items():
        buffer_reader = get_buffer_reader(input_config, buffer_config)
        input_buffers[input_name] = buffer_reader
    return input_buffers


def _plain_sequence(value, *, field_name: str) -> List:
    """Convert one tensor-like diagnostic field to a JSON-safe flat list."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"diagnostic {field_name} must be a flat sequence")
    result = list(value)
    if any(isinstance(item, (list, tuple, dict)) for item in result):
        raise RuntimeError(f"diagnostic {field_name} must be one-dimensional")
    return result


def _diagnostic_record(experience: Experience) -> Dict:
    """Serialize the response arrays omitted by ``Experience.to_dict``."""

    record = experience.to_dict()
    record["diagnostic_schema_version"] = DIAGNOSTIC_EXPERIENCE_SCHEMA_VERSION
    tokens = _plain_sequence(experience.tokens, field_name="tokens")
    prompt_length = experience.prompt_length
    if (
        isinstance(prompt_length, bool)
        or not isinstance(prompt_length, int)
        or not 0 < prompt_length < len(tokens)
    ):
        raise RuntimeError("diagnostic prompt_length is outside token bounds")
    record["response_token_ids"] = tokens[prompt_length:]
    record["old_logprobs"] = _plain_sequence(
        experience.logprobs, field_name="logprobs"
    )
    if experience.action_mask is not None:
        record["action_mask"] = _plain_sequence(
            experience.action_mask, field_name="action_mask"
        )
    return record


class ExperiencePipeline:
    """
    A class to process experiences.
    """

    def __init__(self, config: Config):
        self.logger = get_logger(f"{config.explorer.name}_experience_pipeline", in_ray_actor=True)
        load_plugins()
        pipeline_config = config.data_processor.experience_pipeline
        buffer_config = config.buffer
        self.require_agemem_action_contract = (
            buffer_config.explorer_input.taskset.default_workflow_type
            == "AgeMem_hotpot_workflow_training"
        )
        self.input_store = self._init_input_storage(pipeline_config, buffer_config)  # type: ignore [arg-type]
        try:
            self.operators = ExperienceOperator.create_operators(pipeline_config.operators)
        except Exception as e:
            self.logger.error(f"Failed to create experience operators: {traceback.format_exc()}")
            raise e
        self._set_algorithm_operators(config.algorithm)
        self.output = get_buffer_writer(
            buffer_config.trainer_input.experience_buffer,  # type: ignore [arg-type]
            buffer_config,
        )

    def _init_input_storage(
        self,
        pipeline_config: ExperiencePipelineConfig,
        buffer_config: BufferConfig,
    ) -> Optional[BufferWriter]:
        """Initialize the input storage if it is not already set."""
        if pipeline_config.save_input:
            if pipeline_config.input_save_path is None:
                raise ValueError("input_save_path must be set when save_input is True.")
            elif is_json_file(pipeline_config.input_save_path):
                return get_buffer_writer(
                    StorageConfig(
                        storage_type=StorageType.FILE,
                        path=pipeline_config.input_save_path,
                        wrap_in_ray=False,
                    ),
                    buffer_config,
                )
            elif is_database_url(pipeline_config.input_save_path):
                return get_buffer_writer(
                    StorageConfig(
                        storage_type=StorageType.SQL,
                        path=pipeline_config.input_save_path,
                        wrap_in_ray=False,
                    ),
                    buffer_config,
                )
            else:
                raise ValueError(
                    f"Unsupported save_input format: {pipeline_config.save_input}. "
                    "Only JSON file path or SQLite URL is supported."
                )
        else:
            return None

    def _set_algorithm_operators(self, algorithm_config: AlgorithmConfig) -> None:
        """Add algorithm-specific operators to the pipeline."""
        from trinity.algorithm import ADVANTAGE_FN, ALGORITHM_TYPE

        algorithm = ALGORITHM_TYPE.get(algorithm_config.algorithm_type)
        if not algorithm.compute_advantage_in_trainer and algorithm_config.advantage_fn:
            advantage_fn_cls = ADVANTAGE_FN.get(algorithm_config.advantage_fn)
            assert (
                advantage_fn_cls is not None
            ), f"AdvantageFn {algorithm_config.advantage_fn} not found."
            assert (
                not advantage_fn_cls.compute_in_trainer()
            ), f"AdvantageFn {algorithm_config.advantage_fn} can only be computed in the trainer, please check your implementation."
            self.operators.append(advantage_fn_cls(**algorithm_config.advantage_fn_args))

    async def prepare(self) -> None:
        await self.output.acquire()
        if self.input_store is not None:
            await self.input_store.acquire()

    async def process(self, exps: List[Experience]) -> Dict:
        """Process a batch of experiences.

        Args:
            exps (List[Experience]): List of experiences to process. These experiences are typically generated by an explorer in one step.

        Returns:
            Dict: A dictionary containing metrics collected during the processing of experiences.
        """
        # Fail before both the optional raw-input sink and the trainer buffer.
        # Existing non-AgeMem Experiences have no action contract and remain
        # unaffected; AgeMem workflows require the contract to remain present.
        validate_on_policy_experiences(
            exps,
            require_contract=getattr(
                self, "require_agemem_action_contract", False
            ),
        )

        if self.input_store is not None:
            await self.input_store.write_async(exps)

        metrics = {}

        # Process experiences through operators
        for operator in self.operators:
            exps, metric = operator.process(exps)
            metrics.update(metric)

        # Operators may attach ActionCreditRecords or otherwise transform the
        # batch. Revalidate the final payload at the actual trainer boundary.
        validate_on_policy_experiences(
            exps,
            require_contract=getattr(
                self, "require_agemem_action_contract", False
            ),
        )

        metrics["experience_count"] = len(exps)

        # Write processed experiences to output buffer
        await self.output.write_async(exps)

        # prefix metrics keys with 'pipeline/'
        result_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                result_metrics[f"pipeline/{key}"] = float(value)

        return result_metrics

    async def persist_diagnostic_input(self, exps: List[Experience]) -> Dict:
        """Persist validated bench Experiences without writing the trainer buffer.

        AgeMem's frozen learning-signal diagnosis runs in ``bench`` mode, so
        these Experiences must be available for post-hoc action/token/logprob
        audits while remaining completely outside the optimization path.
        """

        if self.input_store is None:
            raise RuntimeError(
                "diagnostic Experience persistence requires save_input=true"
            )
        validate_on_policy_experiences(
            exps,
            require_contract=getattr(
                self, "require_agemem_action_contract", False
            ),
        )
        records = [_diagnostic_record(experience) for experience in exps]
        await self.input_store.write_async(records)
        return {"pipeline/diagnostic_experience_count": float(len(exps))}

    async def close(self) -> None:
        try:
            await self.output.release()
        except Exception as e:
            self.logger.error(f"Failed to release output buffer: {e}")
        if self.input_store is not None:
            try:
                await self.input_store.release()
            except Exception as e:
                self.logger.error(f"Failed to release diagnostic input buffer: {e}")
        for operator in self.operators:
            operator.close()
