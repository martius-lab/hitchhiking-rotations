<h1 align="center">
  <br>
  Hitchhiking Rotations
  <br>
</h1>

<h4 align="center">
Code for ICML 2024: <a href="some_ariv_link" target="_blank">"Position Paper: Learning with 3D rotations, a hitchhiker’s guide to SO(3)"</a>.</h4>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#development">Development</a> •
  <a href="#credits">Credits</a>
</p>


# Overview
(repository overview)

![assets/docs/torus_v5.pdf](assets/docs/torus_v5.pdf)
      
# Results
(this may be optional)

# Installation
(virtual environment or just list of dependencies) 
(using git lsf to get datasets and our checkpoints/models)

```shell
git clone git@github.com:martius-lab/hitchhiking-rotations.git
pip3 install -e ./
pip3 install torch torchvision torchaudio
```

# Experiments
List of each experiment as in paper and how to reproduce it

# Development
### Code Formatting
```shell
pip3 install black==23.10
cd hitchhiking_rotations && black --line-length 120 ./
```
### Add License Headers
```shell
pip3 install addheader
# If your are using zsh otherwise remove \
addheader hitchhiking_rotations -t .header.txt -p \*.py --sep-len 79 --comment='#' --sep=' '
```

## TODO
- Add the headers
- Change version to 1.0.0 if done
- Make Logger work (only works if r9 now) and store results
- Experiments to do: Check normalization of quat helpfull ?
- Experiments to do: l2 should be same as chordal for r9 ? 
- Test Geodesic Distance
- Write in general for everything tests ideally
- Add plotting scripts

# Credits

