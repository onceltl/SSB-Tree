# SSBTree

## Description

A well-tuned B+-tree for Persistent Memory Systems. 

Please cite our paper if you use SSB-Tree:

> Tongliang Li, Haixia Wang, Airan Sao, Dongsheng Wang. **SSB-Tree: Making Persistent Memory B+-Trees Crash-Consistent and Concurrent by Lazy-Box.**   _Proceedings of the 36th IEEE International Parallel & Distributed Processing Symposium (IPDPS 2022)_.

**Support**: `SSBTree` supports Insert, Delete, Update, Point Lookup, and Range Scan operations. Each operation works for 64-bit integer keys and values.

**Use Case**: `SSBTree` is suitable to be applied for the applications using persistent memory to enable instant recovery.

#### Desired system configurations

- Ubuntu 18.04.1 LTS

- Compile: cmake, g++-7, gcc-7, c++17

#### Dependencies

##### Install build packages

```
$ sudo apt-get install libtbb-dev libjemalloc-dev
```

##### Install PMDK
```
$ git clone https://github.com/pmem/pmdk.git
$ cd pmdk
$ git checkout tags/1.6
$ make -j
$ cd ..
```

### Build & Run

##### Build

```
$ mkdir build
$ cd build
$ cmake .. //-DREBALANCE=on to enable merge, disabled by default
$ make -j
```



##### Run

```
$ sudo ./example 10000 4 /mnt/pmem

usage: ./example [n] [nthreads] [poolpath]
n: number of keys (integer)
nthreads: number of threads (integer)
poolpath: path of the persistent memory pool
````

## Experiment

We support a wrapper for PiBench to easily verify the performance of SSBTree  with other PM B+-trees.

Please check [PiBench](https://github.com/wangtzh/pibench) for more information.
