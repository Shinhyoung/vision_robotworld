from setuptools import find_packages, setup

package_name = "roboworld_ros_utils"

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
    description="ROS 2 adapter helpers shared by the RoboWorld nodes",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
