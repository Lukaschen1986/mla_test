min_max_sca = lambda x: (x-min(x)) / (max(x)-min(x)+10**-8)
min_max_sca_rev = lambda x: (max(x)-x) / (max(x)-min(x)+10**-8)
