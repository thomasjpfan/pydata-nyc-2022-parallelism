# Parallelism in Numerical Python Libraries

[Link to slides](https://thomasjpfan.github.io/pydata-nyc-2022-parallelism)

Python libraries can compute on multiple CPU cores using a variety of parallel programming interfaces such as multiprocessing, pthreads, or OpenMP. Some libraries use an ahead-of-time compiler like Cython or a just-in-time compiler like Numba to parallelize their computational routines. When many levels of parallelism operate simultaneously, it can result in oversubscription and degraded performance. We will learn how parallelism is implemented and configured in various Python libraries such as NumPy, SciPy, and scikit-learn. Furthermore, we will see how to control these mechanisms for parallelism to avoid oversubscription.

## License

This repo is under the [MIT License](LICENSE).
