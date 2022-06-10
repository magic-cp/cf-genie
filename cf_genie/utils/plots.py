from typing import Optional

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .read_write_files import write_plot


def plot_wordcloud(
        text: str,
        file_name: Optional[str] = None,
        plot_title: Optional[str] = None):
    wordcloud = WordCloud(
        background_color='black',
        width=1200,
        height=800,
        max_words=100,
        random_state=42,
        min_word_length=4
    ).generate(text)

    plt.figure()
    if plot_title:
        plt.title(plot_title)
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')

    if file_name:
        write_plot(file_name, plt)
    else:
        plt.show()


def plot_pie_chart(labels, sizes, file_name: Optional[str] = None, plot_title: Optional[str] = None):
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    if plot_title:
        ax1.set_title(plot_title, pad=20)

    if file_name:
        write_plot(file_name, plt)
    else:
        plt.show()
