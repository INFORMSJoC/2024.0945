[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Centrality of shortest paths: algorithms and complexity results

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper [Centrality of shortest paths: algorithms and complexity results](https://doi.org/10.1287/ijoc.2024.0945) by Johnson Phosavanh and Dmytro Matsypura.

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0945

https://doi.org/10.1287/ijoc.2024.0945.cd

Below is the BibTeX for citing this snapshot of the repository.

```
@misc{ShortestPathCentrality,
  author =        {Johnson Phosavanh and Dmytro Matsypura},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Centrality of shortest paths: algorithms and complexity results}},
  year =          {2025},
  doi =           {10.1287/ijoc.2024.0945.cd},
  url =           {https://github.com/INFORMSJoC/2024.0945},
  note =          {Available for download at https://github.com/INFORMSJoC/2024.0945},
}  
```

## Description

This software implements algorithms to find the most degree and betweenness-central shortest paths in a graph in Python. Specifically, this includes Algorithms 1 and 2 from this paper, and Algorithm 1 from Matsypura et al. (2023), which is referred to as Algorithm MVP in this paper.

## Requirements

This code has been written in Python 3.12, and in addition to the standard Python libraries, requires the following packages:

- `networkx`
- `numpy`
- `pandas`

## Results

Tables 1-7 in the paper show the results of the timing tests with different graphs on Mac Studio (2023) with Apple M2 Ultra and 128 GB of RAM running macOS Ventura 13.6.

## Replicating

To replicate the results in Tables 1-7, do

```
python scripts/timing.py
```

You may choose to enable multiprocessing by setting `use_multiprocessing = True` and specifying the number of threads with the variable `n_jobs`.

## References

Matsypura, D., Veremyev, A., Pasiliao, E. L., Prokopyev, O. A. (2023). Finding the most degree-central walks and paths in a graph: Exact and heuristic approaches. *European Journal of Operational Research*, 308(3):1021–1036.
