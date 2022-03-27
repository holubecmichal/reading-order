import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from e_results import EResultCollection


class ResultsGraphs(object):
    def __init__(self, results: EResultCollection):
        self.results = results
        self.winners = None

    def heatmap(self, title):
        read_lengths = self.results.eresults.read_lengths
        candidate_lengths = self.results.eresults.candidate_lengths
        sum, avg, vmax, vmin = self._get_winners()

        f, (sum_ax, avg_ax) = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(sum, linewidth=0.5, xticklabels=candidate_lengths, yticklabels=read_lengths, ax=sum_ax, vmin=vmin,
                    vmax=vmax)
        sum_ax.set_xlabel('Počet tokenů kandidáta', fontsize=15)
        sum_ax.set_ylabel('Počet tokenů pro inicializaci', fontsize=15)
        sum_ax.set_title('SUM')
        sum_ax.tick_params(labelrotation=0)

        sns.heatmap(avg, linewidth=0.5, xticklabels=candidate_lengths, yticklabels=read_lengths, ax=avg_ax, vmin=vmin,
                    vmax=vmax)
        avg_ax.set_xlabel('Počet tokenů kandidáta', fontsize=15)
        avg_ax.set_ylabel('Počet tokenů pro inicializaci', fontsize=15)
        avg_ax.tick_params(labelrotation=0)

        f.suptitle(title)
        plt.plot()
        plt.show()

    def heatmap_avg(self):
        read_lengths = self.results.eresults.read_lengths
        candidate_lengths = self.results.eresults.candidate_lengths
        sum, avg, vmax, vmin = self._get_winners()
        f, avg_ax = plt.subplots(1)
        sns.heatmap(avg, linewidth=0.5, xticklabels=candidate_lengths, yticklabels=read_lengths, ax=avg_ax, vmin=vmin,
                    vmax=vmax)
        avg_ax.set_xlabel('Počet tokenů kandidáta', fontsize=15)
        avg_ax.set_ylabel('Počet tokenů pro inicializaci', fontsize=15)
        avg_ax.tick_params(labelrotation=0)

        plt.tight_layout()
        plt.plot()
        plt.savefig('token_heatmap_avg.eps', format='eps')

    def heatmap_sum(self):
        read_lengths = self.results.eresults.read_lengths
        candidate_lengths = self.results.eresults.candidate_lengths
        sum, avg, vmax, vmin = self._get_winners()

        f, sum_ax = plt.subplots(1)
        sns.heatmap(sum, linewidth=0.5, xticklabels=candidate_lengths, yticklabels=read_lengths, ax=sum_ax, vmin=vmin,
                    vmax=vmax)
        sum_ax.set_xlabel('Počet tokenů kandidáta', fontsize=15)
        sum_ax.set_ylabel('Počet tokenů pro inicializaci', fontsize=15)

        # skryje stupnici
        f.axes[1].set_visible(False)
        sum_ax.tick_params(labelrotation=0)

        # sns.set(font_scale=5)
        plt.tight_layout()
        plt.plot()
        plt.savefig('token_heatmap_sum.eps', format='eps')


    def tendency(self, title):
        sum, avg, vmax, vmin = self._get_winners()
        sum = torch.tensor(sum)
        avg = torch.tensor(avg)

        f, (read_ax, candidate_ax) = plt.subplots(1, 2, figsize=(15, 6))
        read_ax.set_ylim([vmin, vmax])
        read_ax.set_title('read length')
        candidate_ax.set_ylim([vmin, vmax])
        candidate_ax.set_title('candidate length')

        self._plot_tendency(read_ax, sum.mean(dim=1), avg.mean(dim=1))
        self._plot_tendency(candidate_ax, sum.mean(dim=0), avg.mean(dim=0))
        candidate_ax.set_xlabel('Počet tokenů kandidáta', fontsize=15)
        f.suptitle(title)

        plt.show()

    def tendency_read(self):
        sum, avg, vmax, vmin = self._get_winners()
        sum = torch.tensor(sum)
        avg = torch.tensor(avg)

        f, read_ax = plt.subplots(1)
        read_ax.set_ylim([vmin, vmax])
        # read_ax.set_title('read length')
        self._plot_tendency(read_ax, sum.mean(dim=1), avg.mean(dim=1))
        read_ax.set_xlabel('Počet tokenů pro inicializaci', fontsize=15)
        plt.savefig('token_read_tendency.eps', format='eps')

    def tendency_candidate(self):
        sum, avg, vmax, vmin = self._get_winners()
        sum = torch.tensor(sum)
        avg = torch.tensor(avg)

        f, candidate_ax = plt.subplots(1)
        candidate_ax.set_ylim([vmin, vmax])
        # candidate_ax.set_title('candidate length')
        self._plot_tendency(candidate_ax, sum.mean(dim=0), avg.mean(dim=0))
        candidate_ax.set_xlabel('Počet tokenů kandidáta', fontsize=15)
        plt.savefig('token_candidate_tendency.eps', format='eps')

    def total_result_bar(self, title):
        sum = torch.tensor(self.results.get_sum_histogram()) / len(self.results)
        avg = torch.tensor(self.results.get_avg_histogram()) / len(self.results)

        df = pd.DataFrame({'offset': self.results.eresults.candidate_offsets, 'sum': sum, 'avg': avg})
        random_prob = None

        if (df['offset'] == -1).any():
            # existuje random vzorek
            random_prob = df[df['offset'] == -1]
            df = df[df['offset'] != -1]

        ax = sns.barplot(data=df.melt(id_vars='offset', var_name='op'), x='offset', y='value', hue='op', palette=sns.color_palette())
        ax.set_ylabel('probability')

        if random_prob is not None:
            ax.axhline(random_prob.iloc[0]['sum'], color=sns.color_palette()[0], ls='--')
            ax.axhline(random_prob.iloc[0]['avg'], color=sns.color_palette()[1], ls='--')

        for index, row in df.iterrows():
            max_value = row[['sum', 'avg']].max()
            ax.text(index, max_value + 0.005, round(max_value, 2), color='black', ha="center")

        ax.set_title(title)

        plt.show()

    def bar_area(self, title, read_lengths, candidate_lengths):
        offsets = self.results.eresults.candidate_offsets

        data = []
        for i in read_lengths:
            read = self.results.get_by_read_length(i)

            for j in candidate_lengths:
                read_candidate = read.get_by_candidate_length(j)
                sum_prob = np.array(read_candidate.get_sum_histogram()) / self.results.eresults.samples
                avg_prob = np.array(read_candidate.get_avg_histogram()) / self.results.eresults.samples

                for index, offset in enumerate(offsets):
                    data.append({
                        'read_length': i,
                        'candidate_length': j,
                        'offset': offset,
                        'type': 'sum',
                        'probability': sum_prob[index]
                    })

                    data.append({
                        'read_length': i,
                        'candidate_length': j,
                        'offset': offset,
                        'type': 'avg',
                        'probability': avg_prob[index]
                    })

        df = pd.DataFrame(data)
        df.set_index(['read_length', 'candidate_length', 'offset'])

        g = sns.FacetGrid(df[df['offset'] != -1], row='read_length', col='candidate_length')
        g.map_dataframe(sns.barplot, x='offset', y='probability', hue='type', palette=sns.color_palette())
        g.add_legend()

        if (df['offset'] == -1).any():
            randoms = df[df['offset'] == -1]

            for ri, rl in enumerate(read_lengths):
                for ci, cl in enumerate(candidate_lengths):
                    sum = randoms[(randoms['type'] == 'sum') & (randoms['read_length'] == rl) & (randoms['candidate_length'] == cl)]
                    avg = randoms[(randoms['type'] == 'avg') & (randoms['read_length'] == rl) & (randoms['candidate_length'] == cl)]
                    g.axes[ri, ci].axhline(sum.iloc[0]['probability'], color=sns.color_palette()[0], ls='--')
                    g.axes[ri, ci].axhline(avg.iloc[0]['probability'], color=sns.color_palette()[1], ls='--')



        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(title)

        plt.show()

    def _plot_tendency(self, ax, sum_tensor, avg_tensor):
        xi = list(range(len(self.results.eresults.candidate_lengths)))

        ax.plot(xi, sum_tensor, linestyle='-', marker='o', label='SUM')
        ax.axes.set_xticks(xi, self.results.eresults.candidate_lengths)

        ax.plot(xi, avg_tensor, linestyle='-', marker='o', label='AVG')
        ax.axes.set_xticks(xi, self.results.eresults.candidate_lengths)

        ax.set_ylabel('Průměrná úspěšnost', fontsize=15)
        ax.set_xlabel('Počet tokenů pro inicializaci', fontsize=15)
        ax.legend(fontsize=15)

    def _get_winners(self):
        if self.winners:
            return self.winners

        read_lengths = self.results.eresults.read_lengths
        candidate_lengths = self.results.eresults.candidate_lengths
        vmax = 0
        vmin = 1

        sum = []
        avg = []
        for i in read_lengths:
            read = self.results.get_by_read_length(i)

            sum_cols = []
            avg_cols = []

            for j in candidate_lengths:
                read_candidate = read.get_by_candidate_length(j)
                sum_count = read_candidate.get_sum_histogram()[0]
                avg_count = read_candidate.get_avg_histogram()[0]

                sum_cols.append(sum_count / self.results.eresults.samples)
                avg_cols.append(avg_count / self.results.eresults.samples)

            vmax = max(vmax, max(sum_cols), max(avg_cols))
            vmin = min(vmin, min(sum_cols), min(avg_cols))

            sum.append(sum_cols)
            avg.append(avg_cols)

        self.winners = (sum, avg, vmax, vmin)
        return self.winners