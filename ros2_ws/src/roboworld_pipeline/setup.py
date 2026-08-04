from setuptools import find_packages, setup

package_name = "roboworld_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RoboWorld Vision Team",
    maintainer_email="k3i_ai5@k3i.co.kr",
    description="Index-cycle orchestration and the robot-department output topic",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pipeline_node = roboworld_pipeline.pipeline_node:main",
            "mock_camera_node = roboworld_pipeline.mock_camera_node:main",
            "trigger_node = roboworld_pipeline.trigger_node:main",
        ],
    },
)
