from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def uq(experiment_id: str, outdir_root: str) -> None:
    out_parent = Path(outdir_root)
    if not out_parent.exists():
        raise FileNotFoundError(f"The directory {outdir_root} does not exist")

    exp_dir = out_parent / experiment_id
    pq_root_dir = exp_dir / "history" / f"experiment_id={experiment_id}"


def main() -> None:
    app()


if __name__ == "__main__":
    main()
