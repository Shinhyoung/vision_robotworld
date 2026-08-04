from glob import glob

from setuptools import find_packages, setup

package_name = "roboworld_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        # CAD meshes are installed alongside the config so a deployed system
        # never depends on the source tree layout (roboworld_core.paths looks
        # here first via ament, then falls back to 01_input/).
        ("share/" + package_name + "/meshes", glob("meshes/*.ply")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RoboWorld Vision Team",
    maintainer_email="k3i_ai5@k3i.co.kr",
    description="Launch files, configuration and CAD assets for the RoboWorld pipeline",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
