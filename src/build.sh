#!/bin/sh

nohup docker build -t ros-pytorch-2:gpu . >> build.log &
