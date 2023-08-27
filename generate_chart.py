import plotly.graph_objs as go
import datetime
import numpy as np
import plotly.express as px
from plotly.graph_objs import Layout
import plotly.io as pio

# Rozwiązanie problemu wyświetlania w różnych środowiskach
pio.renderers.default = "browser"

with open('rain/rain.txt', 'r') as file:
    data = file.readlines()

x = []
y = []

for i in data:
    line = i.strip().split("\t")
    x.append(line[0])
    y.append(line[1])

y = np.array(y, dtype=float)
avg = np.zeros((len(y), 1))
avg_len = 42

layout = Layout(
        paper_bgcolor='rgb(15,15,15)',
        plot_bgcolor='rgb(23,23,23)',
        # coloraxis='rgb(23,23,23)'
        )
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Temperatura'))
fig.update_xaxes(title_text="Data")
fig.update_yaxes(title_text="Sensor")
fig.show()
print('ok')
