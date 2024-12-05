from setuptools import find_packages, setup

package_name = 'sam2_foundation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aloch',
    maintainer_email='e12044788@student.tuwien.ac.at',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'service = sam2_foundation.segmentation_service:main',
            'action = sam2_foundation.segmentation_action:main',
        ],
    },
)
