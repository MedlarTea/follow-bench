from setuptools import setup, find_packages

# socialRPF — the planning / benchmarking package backing the paper
# "Follow-Bench: A Unified Motion Planning Benchmark for Socially-Aware
# Robot Person Following".
#
# A clean conda env only needs the two `pip install -e .` calls below;
# every runtime dependency (numpy, scipy, cvxpy, osqp, opencv, torch,
# stable-baselines3, treelib, gymnasium, ...) is declared here so the
# eight `*_diff.py` entry scripts under
# `example/robot_person_following/` run out of the box.
#
#     pip install -e socialRPF
#     pip install -e ir-sim
#
# This package bundles code from upstream MIT-licensed projects by
# Han Ruihua (RDA-planner, ir-sim). Provenance and attribution are
# documented in NOTICE.md and LICENSE-RDA-planner; the inner
# `RDA_planner/` Python package keeps its original name to make
# upstream provenance visible at every import site.

setup(
    name='socialRPF',
    version='0.1.0',
    author='Hanjing YE',
    description=(
        'Follow-Bench: A Unified Motion Planning Benchmark for '
        'Socially-Aware Robot Person Following'
    ),
    long_description_content_type='text/markdown',
    license='MIT',
    # `find_packages()` registers every sub-package. The inner
    # `RDA_planner` keeps its original name to credit the upstream
    # ADMM-based RDA solver (https://github.com/hanruihua/RDA-planner).
    packages=find_packages(include=[
        'RDA_planner', 'RDA_planner.*',
        'BSO_HFC_planner', 'BSO_HFC_planner.*',
        'DWA_planner', 'DWA_planner.*',
        'SFM_planner', 'SFM_planner.*',
        'Adap_RPF', 'Adap_RPF.*',
        'traj_predictor', 'traj_predictor.*',
        'follow_ahead_reaction', 'follow_ahead_reaction.*',
    ]),
    python_requires='>=3.9',
    install_requires=[
        # --- MPC / RDA core ---
        'cvxpy==1.5.2',
        'pathos',
        # --- Numerics ---
        # numpy is pinned to match the packaged RL checkpoints in
        # `follow_ahead_reaction/assets/`, which were saved with numpy 2.x.
        'numpy==2.2.6',
        'scipy>=1.13',
        'osqp',
        'matplotlib',
        'numba>=0.65.1',
        # --- Curve generation / IO / utilities ---
        'gctl==1.2',
        'pyyaml',
        'imageio',
        'tqdm',
        'sobol-seq>=0.2.0',
        # --- Perception ---
        'opencv-python',
        'scikit-learn',
        'scikit-image',
        'filterpy',
        # --- RL-based planner (follow-ahead reaction) ---
        # torch>=2.3 is required for stable-baselines3 2.8.x compatibility.
        'torch>=2.3,<3',
        'torchvision>=0.18',
        'pytorch_mppi>=0.9.1',
        'stable-baselines3==2.8.0',
        'treelib',
        'gymnasium',
    ],
)
