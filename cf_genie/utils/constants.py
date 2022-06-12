GRAPHS = 'GRAPHS'
MATH = 'MATH'
GEOMETRY = 'GEOMETRY'
OPTIMIZATION = 'OPTIMIZATION'
ADHOC = 'ADHOC'
SEARCH = 'SEARCH'

TAG_GROUP_MAPPER = {
    'implementation': ADHOC,
    'data structures': ADHOC,
    'constructive algorithms': ADHOC,
    'brute force': ADHOC,
    'sortings': ADHOC,
    'strings': ADHOC,
    'bitmasks': ADHOC,
    'two pointers': ADHOC,
    'hashing': ADHOC,
    'interactive': ADHOC,
    'string suffix structures': ADHOC,
    'binary search': SEARCH,
    'ternary search': SEARCH,
    'meet-in-the-middle': SEARCH,
    'geometry': GEOMETRY,
    'graphs': GRAPHS,
    'dfs and similar': GRAPHS,
    'trees': GRAPHS,
    'dsu': GRAPHS,
    'shortest paths': GRAPHS,
    'graph matchings': GRAPHS,
    'math': MATH,
    'number theory': MATH,
    'combinatorics': MATH,
    'probabilities': MATH,
    'matrices': MATH,
    'fft': MATH,
    'chinese remainder theorem': MATH,
    'greedy': OPTIMIZATION,
    'dp': OPTIMIZATION,
    'divide and conquer': OPTIMIZATION,
    'games': OPTIMIZATION,
    'flows': OPTIMIZATION,
    '2-sat': OPTIMIZATION,
}

TAG_GROUPS = set(TAG_GROUP_MAPPER.values())
