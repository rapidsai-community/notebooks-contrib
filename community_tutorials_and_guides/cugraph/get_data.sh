#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo Downloading ...
wget https://data.rapids.ai/cugraph/benchmark/twitter-2010.csv.gz

echo Decompressing ...
gunzip twitter-2010.csv.gz
