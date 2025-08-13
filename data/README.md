# Notes on data

The data sets used in this paper are given in this directory. This directory is structured as follows:

```
data
│   README.md  
├───raw
│   └───...
├───unweighted
│   ├───directed
│   │   └───real-instances
│   │       └───...
│   └───undirected
│       ├───barabasi-albert
│       │   ├───100-barabasi-albert
│       │   │   └───...
│       │   └───...
│       ├───real-instances
│       │   └───...
│       └───watts-strogatz
│           ├───100-ws-10
│           │   └───...
│           ├───100-ws-20
│           │   └───...
│           └───...
└───weighted
    ├───directed
    │   └───real-instances
    │       └───...
    └───undirected
        └───real-instances
            └───...
```

In this file, we provide information on how the graphs used in this paper were obtained.

## Synthetic graphs

All Barabasi-Albert and Watts-Strogatz graphs were generated using NetworkX (Hagberg et al., 2008).

These graphs can be found in the `unweighted/undirected/barabasi_albert` and `unweighted/undirected/watts-strogatz` subdirectories.

## Real instances

These graphs can be found in the `xxxweighted/xxxdirected/real-instances` subdirectories.

- Krebs: The 9/11 terrorist network (Krebs, 2002).
- Dolphins: Social network of bottlenose dolphins (Lusseau et al., 2003).
- Sandi Auths: An academic collaboration network (Rossi and Ahmed, 2015).
- IEEE Bus: A subnetwork of the US Electric Power System in 1962 (Davis and Hu, 2011).
- Santa Fe: A collaboration network for the Santa Fe Institute (Girvan and Newman, 2002).
- US Air 97: A transportation network of US Air in 1997 (Batagelj and Mrvar, 2006; Davis and Hu, 2011).
- Bus: Bus power system network (Davis and Hu, 2011).
- Email: Network of e-mail interchanges (Guimera et al., 2003; Davis and Hu, 2011).
- Cerevisiae: Biological network of yeast protein-protein interactions (Davis and Hu, 2011).
- Copenhagen calls and sms: Social networks of university students with timestamped data on calls made and SMS messages sent (Sapiezynski et al., 2019).
- Bitcoin Alpha and Bitcoin OTC: Trust scores given by users of other users on the Bitcoin Alpha and Bitcoin OTC platforms (Kumar et al., 2016)
- Advogato: Trust scores given by users of other users on the Advogato community (Massa et al., 2009).

## Preprocessing of instances

The original graphs, as downloaded, are provided in the directory `raw`. We describe how they have been transformed below.

### Copenhagen networks

The sms dataset initially contains information on the edges and timestamps of when the messages were sent. The processed datasets contain information on the number of sms messages sent. The undirected graph was computed by summing the directed edges.

The calls dataset initially contains information on edges, timestamps, and the duration of calls made. This was processed in a similar manner to the sms networks to only contain information on the number of calls made.

### Trust graphs

The Bitcoin Alpha and Bitcoin OTC networks originally contain trust scores in the interval $[-10, 10]$ and timestamps. To obtain the weighted trust scores for our paper, we shift these scores to the interval $[1, 21]$.

The Advogato graph initially had trust scores of $\{0.6, 0.8, 1.0\}$, corresponding to apprentice, journeyer, and master, which we have rescaled to $\{3, 4, 5\}$.

## References

Batagelj V., Mrvar, A. (2006). Pajek datasets.

Davis, T. A., Hu, Y. (2011). The University of Florida Sparse Matrix Collection. *ACM Transactions on Mathematical Software (TOMS)*. 38(1):1-25.

Girvan, M., Newman, M. E. (2002). Community structure in social and biological networks. *Proceedings of the National Academy of Sciences*. 99(12):7821-7826.

Guimera, R., Danon, L., Diaz-Guilera, A., Giralt, F., Arenas, A. (2003). Self-similar community structure in a network of human interactions. *Physical Review E*. 68, 065103

Hagberg, A. A., Schult, D. A., Swart, P. J. (2008), Exploring network structure, dynamics, and function using NetworkX, *Proceedings of the 7th Python in Science Conference (SciPy2008)*, Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA). 11–15.

Krebs, V. (2002), Mapping networks of terrorist cells. *Connections*. 24(3):43-52.

Kumar, S., Spezzano, F., Subrahmanian, V., Faloutsos, C. (2016). Edge weight prediction in weighted signed networks. *2016 IEEE 16th International Conference on Data Mining (ICDM)*. 221–230.

Lusseau, D., Schneider, K., Boisseau, O. J., Haase, P., Slooten, E., & Dawson, S. M. (2003). The bottlenose dolphin community of Doubtful Sound features a large proportion of long-lasting associations - Can geographic isolation explain this unique trait? *Behavioral Ecology and Sociobiology. 54:396-405.

Massa, P., Salvetti, M., Tomasoni, D. (2009). Bowling alone and trust decline in social network sites. *2009 Eighth IEEE International Conference on Dependable, Autonomic and Secure Computing*, 658–663.

Rossi, R., Ahmed, N. (2015). The network data repository with interactive graph analytics and visualization. *Proceedings of the AAAI Conference on Artificial Intelligence*, 29(1).

Sapiezynski, P., Stopczynski, A., Lassen, D. D., Lehmann, S. (2019). Interaction data from the Copenhagen Networks Study. *Scientific Data*, 6:10.
