# On the design of scalable, high-precision spherical-radial Fourier features


Approximation using Fourier features is a popular technique for scaling kernel methods to large-scale problems, with myriad applications in machine learning and statistics. 
This method replaces the integral representation of a shift-invariant kernel with a sum using a quadrature rule. 
The design of the latter is meant to reduce the number of features required for high-precision approximation. 
Specifically, for the squared exponential kernel, one must design a quadrature rule that approximates the Gaussian measure on $\mathbb{R}^d$. 
Previous efforts in this line of research have faced difficulties in higher dimensions. 
We introduce a new family of quadrature rules that accurately approximate the Gaussian measure in higher dimensions by exploiting its isotropy. 
These rules are constructed as a tensor product of a radial quadrature rule and a spherical quadrature rule. 
Compared to previous work, our approach leverages a thorough analysis of the approximation error, which suggests natural choices for both the radial and spherical components. 
We demonstrate that this family of Fourier features yields improved approximation bounds.
