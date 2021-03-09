---
aspectratio: 169
fontfamily: bookman
---

# Simple usage

<!--
Build with:
pandoc docs/banner_slides.md -t beamer -o banner_slides.pdf

Converted to GIF with ezgif.com, 300 ms delay time.
-->

```python
import boost_histogram as bh

# Make a histogram
h = bh.Histogram(
    bh.axis.Regular(10, 0, 1)
)

# Fill it with events
h.fill([.2, .3, .6, .9])

# Compute the sum
total = h.sum()
```


# Advanced Indexing

```python
# Slice in data coordinates
sliced_h = h[bh.loc(.5):bh.loc(1.5)]

# Sum over and rebin easily
smaller = h[::sum, ::bh.rebin(2)]

# Set and access easily
h[...] = np.asarray(prebinned)

# Project and reorder axes
h2 = h.project(1, 0)
```

# Many Axis types
::: columns
::: {.column width="60%"}

* Regular: evenly or functionally spaced binning
* Variable: arbitrary binning
* Integer: integer binning
* IntCategory: arbitrary integers
* StrCategory: strings
* Boolean: True/False

Most Axis types support optional extras:

* Underflow and/or Overflow bins
* Growth
* Circular (wraparound)

:::
::: {.column width="40%"}

\begin{center}
  \includegraphics[width=1.05\textwidth]{docs/_images/axis_circular.png}
  \includegraphics[width=.7\textwidth]{docs/_images/axis_regular.png}
  \includegraphics[width=.9\textwidth]{docs/_images/axis_variable.png}
  \includegraphics[width=.65\textwidth]{docs/_images/axis_integer.png}
  \includegraphics[width=\textwidth]{docs/_images/axis_category.png}
\end{center}

:::
:::



# Advanced Storages

* Integers
* Floats
* Weighted Sums
* Means
* Weighted Means

Supports the UHI `PlottableHistogram` protocol!

```python
import mplhep
mplhep.histplot(h)
```



# NumPy compatibility

```python
# Drop in faster replacement
H, edges = bh.numpy.histogram(data)

# Drop in threaded faster replacement
H, edges = bh.numpy.histogram(data, threads=4)

# Drop in much faster replacement for 2D+
H, e1, e2 = bh.numpy.histogram2d(xdata, ydata)

# Return a Histogram, convert back to NumPy style
h = bh.numpy.histogram(data, histogram=bh.Histogram)
H, edges = h.to_numpy()
```
