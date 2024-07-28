import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd

# Set default theme to 'plotly_dark'
pio.templates.default = "plotly_dark"

def plot(scores, mean_scores):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(x=list(range(len(scores))), y=scores, mode='lines+markers', name='Scores'))
    fig.add_trace(go.Scatter(x=list(range(len(mean_scores))), y=mean_scores, mode='lines+markers', name='Mean Scores'))

    fig.update_layout(title='Training Progress',
                      xaxis_title='Number of Games',
                      yaxis_title='Score',
                      template='plotly_dark')

    fig.show()
