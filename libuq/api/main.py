import logging
from contextlib import asynccontextmanager
from pathlib import Path

import fastapi
import uvicorn
from fastapi import APIRouter, FastAPI

from libuq import handlers
from libuq.api.models import SystemConfig

logger = logging.getLogger()


@asynccontextmanager
async def lifecycle(_app):
    try:
        logger.info("API LIFECYCLE INIT")
        yield _app
    finally:
        logger.info("API LIFECYCLE CLOSE")


api_router = APIRouter(prefix="/api")


@api_router.post(
    "/pipeline/samples",
    operation_id="generate-samples",
    description=(
        """ \
        Generate Latin Hypercube Samples for UQ sensitivity analysis.

        By default, varies 5 physiologically relevant sim_data parameters.
        Use --params-file to specify custom parameters via a JSON file.
        Use --include-vio / --include-mecillinam for legacy vio/mecillinam mode.
    """
    ),
)
def generate_samples(
    experiment_ids: list[str],
    system_config: SystemConfig,
    sim_base_path: str | None = None,
    cache_dir: str | None = None,
    n_samples: int = 200,
    seed: int = 42,
    max_workers: int | None = None,
    max_duration: float = 10800.0,
    generations: int = 1,
    live: bool = True,
    batch_dir: str | None = None,
) -> None:
    """Generate LHS samples, evaluate simulation function, cache (X, Y).

    By default varies 5 physiologically relevant scalar sim_data
    parameters.  Pass --params-file to specify custom parameters.
    Pass --include-vio / --include-mecillinam for legacy mode.
    Pass --live to run real vEcoli simulations as subprocesses.
    """
    handlers.verify_out_dirs(sim_base_path, experiment_ids)

    if batch_dir is not None:
        batch_dir = Path(batch_dir)

    samples = handlers.generate_samples(
        experiment_ids=experiment_ids,
        system_config=system_config,
        sim_base_path=sim_base_path,
        cache_dir=cache_dir,
        n_samples=n_samples,
        seed=seed,
        max_workers=max_workers,
        max_duration=max_duration,
        generations=generations,
        live=live,
        batch_dir=batch_dir,
    )
    print(samples)


app = FastAPI(title="uq-api", lifespan=lifecycle)

app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8888)
