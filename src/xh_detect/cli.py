import typer

from xh_detect import __version__

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(f"xh-detect {__version__}")


if __name__ == "__main__":
    app()
