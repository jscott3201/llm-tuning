"""Shared QAT image declaration; importing it creates no App or Volume."""
from pathlib import Path

IMAGE = (
    "docker.io/vllm/vllm-openai:v0.30.0-cu129@"
    "sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688"
)

# Modal discovers `python` on PATH. Reuse the image's system interpreter so
# its preinstalled serving packages stay in the same environment.
PYTHON_SETUP = [
    "RUN test -x /usr/bin/python3.12 && "
    "if [ ! -e /usr/local/bin/python ] && [ ! -L /usr/local/bin/python ]; then "
    "ln -s /usr/bin/python3.12 /usr/local/bin/python; fi && "
    'test "$(readlink -f /usr/local/bin/python)" = "$(readlink -f /usr/bin/python3.12)"',
    "RUN command -v pip && python -m pip --version && "
    "python -c \"import os, sys; from importlib.metadata import version; "
    "from importlib.util import find_spec; "
    "assert os.path.realpath(sys.executable) == '/usr/bin/python3.12'; "
    "assert sys.version_info[:2] == (3, 12); assert sys.prefix == sys.base_prefix; "
    "assert version('vllm').split('+', 1)[0] == '0.30.0'; "
    "print('python', sys.executable, sys.version, sys.prefix); "
    "print('vllm', version('vllm'), find_spec('vllm').origin); "
    "print('torch', version('torch'), find_spec('torch').origin)\"",
]


def build_image():
    """Declare the CPU repair/guard layers before any GPU function is started."""
    import modal

    source = Path(__file__).resolve().parent
    return (
        modal.Image.from_registry(IMAGE, setup_dockerfile_commands=PYTHON_SETUP)
        .entrypoint([])
        .add_local_file(source / "qat_repair.py", "/opt/qat/qat_repair.py", copy=True)
        .add_local_file(source / "qat_stack.py", "/opt/qat/qat_stack.py", copy=True)
        .run_commands("/usr/bin/python3.12 /opt/qat/qat_repair.py",
                      "/usr/bin/python3.12 /opt/qat/qat_stack.py", gpu=None)
        .add_local_python_source("_common")
    )
