"""Behavioral fail-closed tests that do not require Ray or a model runtime."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
from uuid import uuid4

from trinity.common.runtime_receipt import write_training_receipt


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures"


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@contextmanager
def _load_source(relative_path: str, stubs: dict[str, ModuleType]):
    module_name = f"_m8b_runtime_{Path(relative_path).stem}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs, clear=False):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(module_name, None)


class _Logger:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.infos.append(message)


def _workflow_runner_stubs() -> dict[str, ModuleType]:
    return {
        "trinity.buffer": _module(
            "trinity.buffer", get_buffer_reader=lambda *_args, **_kwargs: None
        ),
        "trinity.common.action_event_contract": _module(
            "trinity.common.action_event_contract",
            ActionContractError=type("ActionContractError", (RuntimeError,), {}),
            finalize_experience_action_contract=lambda *_args, **_kwargs: None,
            freeze_rollout_policy_version=lambda first, second: first
            if first == second
            else second,
        ),
        "trinity.common.config": _module("trinity.common.config", Config=object),
        "trinity.common.experience": _module(
            "trinity.common.experience", Experience=object
        ),
        "trinity.common.models": _module(
            "trinity.common.models",
            get_debug_inference_model=lambda _config: (None, []),
        ),
        "trinity.common.models.model": _module(
            "trinity.common.models.model", InferenceModel=object, ModelWrapper=object
        ),
        "trinity.common.workflows": _module(
            "trinity.common.workflows", Task=object, Workflow=object
        ),
        "trinity.utils.log": _module(
            "trinity.utils.log", get_logger=lambda *_args, **_kwargs: _Logger()
        ),
    }


def _explorer_stubs() -> dict[str, ModuleType]:
    ray = _module("ray")
    ray.actor = SimpleNamespace(ActorHandle=object)
    ray.remote = lambda value: value
    ray.get_runtime_context = lambda: SimpleNamespace(node_id="node", namespace="ns")
    ray_util = _module("ray.util", get_node_ip_address=lambda: "127.0.0.1")
    scheduling = _module(
        "ray.util.scheduling_strategies",
        NodeAffinitySchedulingStrategy=object,
    )
    ray.util = ray_util
    torch = _module("torch", Tensor=object)

    constants = SimpleNamespace(
        ROLLOUT_WEIGHT_SYNC_GROUP_NAME="rollout",
        RunningStatus=SimpleNamespace(
            REQUIRE_SYNC="require_sync", RUNNING="running", STOPPED="stopped"
        ),
        SyncMethod=SimpleNamespace(NCCL="nccl"),
        SyncStyle=SimpleNamespace(
            FIXED="fixed", DYNAMIC_BY_EXPLORER="dynamic_by_explorer"
        ),
    )
    return {
        "ray": ray,
        "ray.util": ray_util,
        "ray.util.scheduling_strategies": scheduling,
        "torch": torch,
        "trinity.buffer.buffer": _module(
            "trinity.buffer.buffer", get_buffer_reader=lambda *_args, **_kwargs: None
        ),
        "trinity.buffer.pipelines.experience_pipeline": _module(
            "trinity.buffer.pipelines.experience_pipeline", ExperiencePipeline=object
        ),
        "trinity.common.config": _module("trinity.common.config", Config=object),
        "trinity.common.constants": _module(
            "trinity.common.constants", **vars(constants)
        ),
        "trinity.common.models": _module(
            "trinity.common.models", create_inference_models=lambda _config: ([], [])
        ),
        "trinity.common.models.utils": _module(
            "trinity.common.models.utils",
            get_checkpoint_dir_with_step_num=lambda *_args, **_kwargs: (None, 0),
        ),
        "trinity.explorer.scheduler": _module(
            "trinity.explorer.scheduler", Scheduler=object
        ),
        "trinity.manager.state_manager": _module(
            "trinity.manager.state_manager", StateManager=object
        ),
        "trinity.manager.synchronizer": _module(
            "trinity.manager.synchronizer", Synchronizer=object
        ),
        "trinity.utils.annotations": _module(
            "trinity.utils.annotations", Experimental=lambda function: function
        ),
        "trinity.utils.log": _module(
            "trinity.utils.log", get_logger=lambda *_args, **_kwargs: _Logger()
        ),
        "trinity.utils.monitor": _module(
            "trinity.utils.monitor",
            MONITOR=SimpleNamespace(),
            gather_metrics=lambda *_args, **_kwargs: {},
        ),
        "trinity.utils.plugin_loader": _module(
            "trinity.utils.plugin_loader", load_plugins=lambda: None
        ),
    }


def _trainer_stubs() -> dict[str, ModuleType]:
    class _Timer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    return {
        "pandas": _module("pandas", DataFrame=lambda value: value),
        "ray": _module("ray", remote=lambda value: value),
        "trinity.algorithm": _module(
            "trinity.algorithm", SAMPLE_STRATEGY=SimpleNamespace(get=lambda _name: None)
        ),
        "trinity.common.config": _module("trinity.common.config", Config=object),
        "trinity.common.constants": _module(
            "trinity.common.constants",
            RunningStatus=SimpleNamespace(RUNNING="running", STOPPED="stopped"),
            SyncMethod=SimpleNamespace(
                NCCL="nccl", CHECKPOINT="checkpoint", MEMORY="memory"
            ),
            SyncStyle=SimpleNamespace(),
        ),
        "trinity.common.experience": _module(
            "trinity.common.experience", Experiences=list
        ),
        "trinity.manager.state_manager": _module(
            "trinity.manager.state_manager", StateManager=object
        ),
        "trinity.manager.synchronizer": _module(
            "trinity.manager.synchronizer", Synchronizer=object
        ),
        "trinity.utils.log": _module(
            "trinity.utils.log", get_logger=lambda *_args, **_kwargs: _Logger()
        ),
        "trinity.utils.monitor": _module(
            "trinity.utils.monitor", MONITOR=SimpleNamespace()
        ),
        "trinity.utils.plugin_loader": _module(
            "trinity.utils.plugin_loader", load_plugins=lambda: None
        ),
        "trinity.utils.timer": _module("trinity.utils.timer", Timer=_Timer),
    }


class WorkflowRunnerFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_provider_error_is_returned_as_opaque_failed_status(self) -> None:
        with _load_source(
            "trinity/explorer/workflow_runner.py", _workflow_runner_stubs()
        ) as module:
            secret = "arbitrary provider body must not leak"
            runner = module.WorkflowRunner.__new__(module.WorkflowRunner)
            runner.logger = _Logger()

            class _ModelWrapper:
                @property
                def model_version_async(self):
                    async def _version():
                        return 0

                    return _version()

            runner.model_wrapper = _ModelWrapper()
            runner._run_task = mock.AsyncMock(
                side_effect=RuntimeError(secret)
            )

            status, experiences = await runner.run_task(SimpleNamespace())

            self.assertFalse(status.ok)
            self.assertEqual(experiences, [])
            self.assertEqual(status.message, "Workflow task failed (RuntimeError)")
            self.assertNotIn(secret, status.message)
            self.assertNotIn(secret, "\n".join(runner.logger.errors))


class ExplorerFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_explore_loop_propagates_failure_without_logging_body(self) -> None:
        with _load_source("trinity/explorer/explorer.py", _explorer_stubs()) as module:
            secret = "arbitrary explorer body must not leak"
            explorer = module.Explorer.__new__(module.Explorer)
            explorer.explore_step_num = 0
            explorer.config = SimpleNamespace(
                explorer=SimpleNamespace(name="explorer")
            )
            explorer.logger = _Logger()
            explorer.explore_step = mock.AsyncMock(
                side_effect=RuntimeError(secret)
            )

            with self.assertRaises(RuntimeError):
                await explorer.explore()

            logged = "\n".join(explorer.logger.errors)
            self.assertIn("RuntimeError", logged)
            self.assertNotIn(secret, logged)

    async def test_failed_rollout_is_never_processed(self) -> None:
        with _load_source("trinity/explorer/explorer.py", _explorer_stubs()) as module:
            explorer = module.Explorer.__new__(module.Explorer)
            explorer.scheduler = SimpleNamespace(
                get_results=mock.AsyncMock(
                    return_value=([SimpleNamespace(ok=False)], [object()])
                )
            )
            process = mock.Mock()
            explorer.experience_pipeline = SimpleNamespace(
                process=SimpleNamespace(remote=process)
            )

            with self.assertRaisesRegex(RuntimeError, "rollout tasks failed"):
                await explorer._finish_explore_step(step=1, model_version=0)

            process.assert_not_called()

    async def test_failed_eval_never_writes_success_receipt(self) -> None:
        with _load_source("trinity/explorer/explorer.py", _explorer_stubs()) as module:
            explorer = module.Explorer.__new__(module.Explorer)
            explorer.pending_eval_tasks = deque([(1, "heldout")])
            explorer.scheduler = SimpleNamespace(
                get_results=mock.AsyncMock(
                    return_value=(
                        [SimpleNamespace(ok=False, metric={})],
                        [],
                    )
                )
            )
            explorer.explore_step_num = 1
            explorer.model_version = 0
            explorer.config = SimpleNamespace(checkpoint_job_dir="unused")
            explorer.monitor = SimpleNamespace(log=mock.Mock())
            module.write_benchmark_receipt = mock.Mock()

            with self.assertRaisesRegex(RuntimeError, "evaluation tasks failed"):
                await explorer._finish_eval_step(step=1, prefix="bench")

            module.write_benchmark_receipt.assert_not_called()
            explorer.monitor.log.assert_not_called()

    async def test_base_benchmark_records_model_version_zero(self) -> None:
        with _load_source("trinity/explorer/explorer.py", _explorer_stubs()) as module:
            explorer = module.Explorer.__new__(module.Explorer)
            explorer.config = SimpleNamespace(
                explorer=SimpleNamespace(
                    bench_on_latest_checkpoint=False,
                    eval_on_startup=True,
                ),
                checkpoint_job_dir=str(FIXTURE_ROOT),
            )
            explorer.model_version = -1

            async def _finish(*_args, **_kwargs):
                self.assertEqual(explorer.model_version, 0)

            explorer._finish_eval_step = _finish

            self.assertTrue(await explorer.benchmark())

    async def test_checkpoint_benchmark_records_loaded_model_version(self) -> None:
        with _load_source("trinity/explorer/explorer.py", _explorer_stubs()) as module:
            explorer = module.Explorer.__new__(module.Explorer)
            explorer.config = SimpleNamespace(
                explorer=SimpleNamespace(
                    bench_on_latest_checkpoint=True,
                    eval_on_startup=False,
                )
            )
            explorer.model_version = -1
            explorer._checkpoint_weights_update = mock.AsyncMock(return_value=1)
            explorer.eval = mock.AsyncMock()

            async def _finish(*_args, **_kwargs):
                self.assertEqual(explorer.explore_step_num, 1)
                self.assertEqual(explorer.model_version, 1)

            explorer._finish_eval_step = _finish

            self.assertTrue(await explorer.benchmark())
            explorer.eval.assert_awaited_once()


class TrainerFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_training_receipt_includes_batch_reward_metrics(self) -> None:
        with _load_source("trinity/trainer/trainer.py", _trainer_stubs()) as module:
            class _Scalar:
                def __init__(self, value: float) -> None:
                    self.value = value

                def item(self) -> float:
                    return self.value

            class _Rewards:
                def mean(self):
                    return _Scalar(0.5)

                def min(self):
                    return _Scalar(0.0)

                def max(self):
                    return _Scalar(1.0)

            engine = SimpleNamespace(train_step_num=0)
            trainer = module.Trainer.__new__(module.Trainer)
            trainer.engine = engine
            trainer.total_steps = 1
            trainer.config = SimpleNamespace(
                checkpoint_job_dir="unused",
                trainer=SimpleNamespace(name="trainer", enable_preview=False),
            )
            trainer.logger = _Logger()
            exps = SimpleNamespace(rewards=_Rewards())
            trainer._sample_data = mock.AsyncMock(return_value=(exps, {}, []))
            trainer.need_sync = mock.AsyncMock(return_value=False)

            async def _train_step(_exps):
                engine.train_step_num = 1
                return {
                    "actor/loss": 0.25,
                    "actor/ppo_kl": 0.01,
                    "training/actor_update_completed": 1.0,
                }

            trainer.train_step = _train_step
            trainer.need_save = mock.Mock(return_value=False)
            trainer.save_checkpoint = mock.Mock(return_value={})
            trainer.monitor = SimpleNamespace(log=mock.Mock())
            set_status = mock.AsyncMock()
            trainer.synchronizer = SimpleNamespace(
                set_trainer_status=SimpleNamespace(remote=set_status)
            )
            module.write_training_receipt = mock.Mock()

            await trainer.train()

            receipt_metrics = module.write_training_receipt.call_args.kwargs[
                "metrics"
            ]
            self.assertEqual(receipt_metrics["training/reward_mean"], 0.5)
            self.assertEqual(receipt_metrics["training/reward_min"], 0.0)
            self.assertEqual(receipt_metrics["training/reward_max"], 1.0)
            trainer.save_checkpoint.assert_called_once_with(
                block_until_saved=True,
                save_as_hf=True,
            )

    async def test_train_step_failure_propagates_without_receipt_or_checkpoint(self) -> None:
        with _load_source("trinity/trainer/trainer.py", _trainer_stubs()) as module:
            secret = "arbitrary trainer body must not leak"
            trainer = module.Trainer.__new__(module.Trainer)
            trainer.engine = SimpleNamespace(train_step_num=0)
            trainer.total_steps = 1
            trainer.config = SimpleNamespace(
                checkpoint_job_dir="unused",
                trainer=SimpleNamespace(name="trainer", enable_preview=False),
            )
            trainer.logger = _Logger()
            trainer._sample_data = mock.AsyncMock(return_value=([], {}, []))
            trainer.need_sync = mock.AsyncMock(return_value=False)
            trainer.train_step = mock.AsyncMock(
                side_effect=RuntimeError(secret)
            )
            trainer.save_checkpoint = mock.Mock()
            module.write_training_receipt = mock.Mock()

            with self.assertRaises(RuntimeError):
                await trainer.train()

            module.write_training_receipt.assert_not_called()
            trainer.save_checkpoint.assert_not_called()
            logged = "\n".join(trainer.logger.errors)
            self.assertIn("RuntimeError", logged)
            self.assertNotIn(secret, logged)

    async def test_finite_input_exhaustion_is_an_error(self) -> None:
        with _load_source("trinity/trainer/trainer.py", _trainer_stubs()) as module:
            trainer = module.Trainer.__new__(module.Trainer)
            trainer.engine = SimpleNamespace(train_step_num=0)
            trainer.total_steps = 1
            trainer.config = SimpleNamespace(
                checkpoint_job_dir="unused",
                trainer=SimpleNamespace(name="trainer", enable_preview=False),
            )
            trainer.logger = _Logger()
            trainer._sample_data = mock.AsyncMock(side_effect=StopAsyncIteration)
            trainer.need_sync = mock.AsyncMock(return_value=False)
            trainer.save_checkpoint = mock.Mock()

            with self.assertRaisesRegex(RuntimeError, "0/1"):
                await trainer.train()

            trainer.save_checkpoint.assert_not_called()

    async def test_failed_nccl_sync_does_not_advance_or_report_running(self) -> None:
        with _load_source("trinity/trainer/trainer.py", _trainer_stubs()) as module:
            trainer = module.Trainer.__new__(module.Trainer)
            trainer.engine = SimpleNamespace(train_step_num=1, sync_weight=mock.Mock())
            trainer.config = SimpleNamespace(
                synchronizer=SimpleNamespace(sync_method="nccl")
            )
            trainer.logger = _Logger()
            trainer.last_sync_step = None
            trainer.last_trainer_sync_step = 0
            set_status = mock.AsyncMock()
            trainer.synchronizer = SimpleNamespace(
                ready_to_nccl_sync=SimpleNamespace(
                    remote=mock.AsyncMock(return_value=None)
                ),
                set_trainer_status=SimpleNamespace(remote=set_status),
            )

            with self.assertRaisesRegex(RuntimeError, "failed before NCCL"):
                await trainer.sync_weight()

            trainer.engine.sync_weight.assert_not_called()
            set_status.assert_not_awaited()
            self.assertIsNone(trainer.last_sync_step)
            self.assertEqual(trainer.last_trainer_sync_step, 0)


class LauncherFailureTest(unittest.TestCase):
    def test_both_observes_other_actor_failure_after_trainer_finishes(self) -> None:
        secret = "arbitrary launcher body must not leak"
        logger = _Logger()

        class _RemoteMethod:
            def __init__(self, token: str) -> None:
                self.token = token

            def remote(self):
                return self.token

        class _Actor:
            def __init__(self, prefix: str) -> None:
                setattr(self, "__ray_ready__", _RemoteMethod(f"{prefix}-ready"))
                self.prepare = _RemoteMethod(f"{prefix}-prepare")
                self.sync_weight = _RemoteMethod(f"{prefix}-sync")
                self.explore = _RemoteMethod("explore-run")
                self.train = _RemoteMethod("train-run")
                self.shutdown = _RemoteMethod(f"{prefix}-shutdown")

        explorer = _Actor("explorer")
        trainer = _Actor("trainer")
        ray = _module("ray")

        def _get(value):
            if value == "train-run":
                return "trainer"
            if value == ["explore-run"]:
                raise RuntimeError(secret)
            return None

        waits = iter(
            [
                (["train-run"], ["explore-run"]),
                (["explore-run"], []),
            ]
        )
        ray.get = _get
        ray.wait = lambda *_args, **_kwargs: next(waits)
        ray.init = lambda *_args, **_kwargs: None
        ray.shutdown = lambda: None

        class _Explorer:
            get_actor = staticmethod(lambda _config: explorer)

        class _Trainer:
            get_actor = staticmethod(lambda _config: trainer)

        stubs = {
            "ray": ray,
            "trinity.buffer.pipelines.task_pipeline": _module(
                "trinity.buffer.pipelines.task_pipeline",
                check_and_run_task_pipeline=lambda _config: None,
            ),
            "trinity.common.config": _module(
                "trinity.common.config", Config=object, load_config=lambda _path: None
            ),
            "trinity.common.constants": _module(
                "trinity.common.constants",
                DEBUG_NAMESPACE="debug",
                PLUGIN_DIRS_ENV_VAR="TRINITY_PLUGIN_DIRS",
            ),
            "trinity.explorer.explorer": _module(
                "trinity.explorer.explorer", Explorer=_Explorer
            ),
            "trinity.manager.state_manager": _module(
                "trinity.manager.state_manager", StateManager=object
            ),
            "trinity.trainer.trainer": _module(
                "trinity.trainer.trainer", Trainer=_Trainer
            ),
            "trinity.utils.dlc_utils": _module(
                "trinity.utils.dlc_utils",
                is_running=lambda: True,
                setup_ray_cluster=lambda **_kwargs: None,
                stop_ray_cluster=lambda **_kwargs: None,
            ),
            "trinity.utils.log": _module(
                "trinity.utils.log", get_logger=lambda *_args, **_kwargs: logger
            ),
            "trinity.utils.plugin_loader": _module(
                "trinity.utils.plugin_loader", load_plugins=lambda: None
            ),
        }
        config = SimpleNamespace(
            trainer=SimpleNamespace(name="trainer"),
            explorer=SimpleNamespace(name="explorer"),
            synchronizer=SimpleNamespace(sync_timeout=10),
        )

        with _load_source("trinity/cli/launcher.py", stubs) as module:
            with self.assertRaises(RuntimeError):
                module.both(config)

        logged = "\n".join(logger.errors)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(secret, logged)


class RuntimeReceiptFailureTest(unittest.TestCase):
    def test_nonfinite_training_metric_never_creates_receipt(self) -> None:
        output = FIXTURE_ROOT / f".m8b-runtime-{uuid4().hex}"
        try:
            with self.assertRaisesRegex(ValueError, "not finite"):
                write_training_receipt(
                    str(output),
                    completed_step=1,
                    configured_total_steps=1,
                    metrics={"training/loss": float("nan")},
                )
            self.assertFalse(output.exists())
        finally:
            shutil.rmtree(output, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
