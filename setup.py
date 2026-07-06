from setup import setup, find_packages

def get_requirements(filename):
    requirements = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-e "):
                continue
            requirements.append(line)
    return requirements

setup(
    name="mlproject",
    version="0.1",
    author="Raj Namdev",
    author_email="rajnamdev1902@gmail.com",
    description="A machine learning project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)