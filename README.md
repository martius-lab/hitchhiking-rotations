# Hitchhiking Rotations

<h4 align="center">
Code for ICML 2024 "Position Paper: Learning with 3D rotations, a hitchhiker’s guide to SO(3)" <a href="some_ariv_link" target="_blank">Paper</a>.</h4>

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

<object data="https://github.com/martius-lab/hitchhiking-rotations/blob/main/assets/docs/torus_v5.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/martius-lab/hitchhiking-rotations/blob/main/assets/docs/torus_v5.pdf>
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/martius-lab/hitchhiking-rotations/blob/main/assets/docs/torus_v5.pdf">Download PDF</a>.</p>
    </embed>
</object>

# Results
(this may be optional)

# Installation
(virtual environment or just list of dependencies) 
(using git lsf to get datasets and our checkpoints/models)

```
git clone git@github.com:martius-lab/hitchhiking-rotations.git
pip3 install -e ./
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
pip3 install adheader
# If your are using zsh otherwise remove \
addheader hitchhiking_rotations -t .header.txt -p \*.py --sep-len 79 --comment='#' --sep=' '
```

## TODO
- Add the headers
- Change version to 1.0.0 if done

# Credits

