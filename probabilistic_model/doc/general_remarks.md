# General Remarks


## RuntimeWarning: divide by zero encountered in log
You may get the warning `RuntimeWarning: divide by zero encountered in log`. Don't worry about it.
In probability, as soon as an impossible event is encountered, the probability of that event is defined to be zero.
In the logarithmic scale, this means that the log of zero is defined to be negative infinity.
However, such a warning is raised whenever a finite number transitions to an infinite number.
You can safely ignore this warning or turn it off / on by adding 
```
numpy.seterr(divide = 'ignore') 
numpy.seterr(divide = 'warn') 
```
to your source code. 
Checkout [this discussion](https://github.com/numpy/numpy/issues/19723) to learn more about that behaviour.