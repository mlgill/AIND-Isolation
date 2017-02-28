import re
import pandas as pd

# read data
with open('results.md','r') as fh:
	text = fh.read()

# split sections and heuristic labels
heuristics = re.findall(r'''\#\# (.+)''', text)

chunks = re.split(r'''\#\# .+''', text)[1:]
chunks = [re.findall(r'''Match .+\n''', x) for x in chunks]

split_chunks = [re.findall(r'''Match ([1-7]):\s+(\w+)\s+vs\s+(\w+)\s+Result: (\d+) to (\d+)''', 
                ''.join(x)) for x in chunks]

# create dataframes from sections
cols = ['match','opp1','opp2','wins','losses']
split_df = [pd.DataFrame(x, columns=cols) for x in split_chunks]


for heur,df in zip(heuristics, split_df):
	df['heuristic'] = heur

data = pd.concat(split_df).reset_index(drop=True)
data = data[['heuristic'] + cols]

data['wins'] = data.wins.astype('int')
data['losses'] = data.losses.astype('int')
data['games'] = data.wins + data.losses

# calculate statistics
stats = data.groupby(['heuristic','opp1']).agg({'wins':sum, 'losses':sum})
stats['percent'] = stats.wins.astype('float') / 1400

print(stats)