from itertools import product


import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_runner import get_model_suffix_name_for_all_classes

from sklearn.metrics import f1_score, hamming_loss

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():

    REPORT_CONFIG = [
        # Uncomment to test the imbalanced case
        {
            'test_dataset': 'without-adhoc-test',
            'training_dataset': 'without-adhoc-train-balanced',
            'result_label': r'balanceado sin \adhoc{}',
        },
        {
            'test_dataset': 'test',
            'training_dataset': 'train',
            'result_label': r'con \adhoc',
        }
    ]
    results = {r'con \adhoc': {}, r'balanceado sin \adhoc{}': {}}

    to_percentage = lambda x: f'{x * 100:.0f}\\%'

    for report_config, model_class, embedder_class in product(REPORT_CONFIG, SUPERVISED_MODELS, EMBEDDERS):
        y_true_train = utils.read_cleaned_dataset(report_config['training_dataset'])['most_occurrent_tag_group'].to_numpy()
        y_true_test = utils.read_cleaned_dataset(report_config['test_dataset'])['most_occurrent_tag_group'].to_numpy()
        embedder_training = embedder_class([], label=report_config['training_dataset'], training_label=report_config['training_dataset'])
        embedder_test = embedder_class([], label=report_config['test_dataset'], training_label=report_config['training_dataset'])
        model = model_class(
            embedder_training.read_embedded_words,
            y_true_train,
            label=get_model_suffix_name_for_all_classes(embedder_training.embedder_name))

        log.info('Running model %s', model.model_name)
        y_pred_train = model.predict(embedder_training.read_embedded_words())
        y_pred_test = model.predict(embedder_test.read_embedded_words())

        if model.display_name not in results[report_config['result_label']]:
            results[report_config['result_label']][model.display_name] = {}
        results[report_config['result_label']][model.display_name][embedder_training.display_name] = {
            'entrenamiento': {
                'F1 Micro': to_percentage(f1_score(y_true_train, y_pred_train, average='micro')),
                'F1 Macro': to_percentage(f1_score(y_true_train, y_pred_train, average='macro')),
                'F1 Ponderado': to_percentage(f1_score(y_true_train, y_pred_train, average='weighted')),
                'Pérdida de hamming': to_percentage(hamming_loss(y_true_train, y_pred_train)),
            },
            'prueba': {
                'F1 Micro': to_percentage(f1_score(y_true_test, y_pred_test, average='micro')),
                'F1 Macro': to_percentage(f1_score(y_true_test, y_pred_test, average='macro')),
                'F1 Ponderado': to_percentage(f1_score(y_true_test, y_pred_test, average='weighted')),
                'Pérdida de hamming': to_percentage(hamming_loss(y_true_test, y_pred_test)),
            }
        }

    def make_table_for_metric_appendix(metric, data_source, label):
        for run_modality in results:
            print(r'\begin{longtable}[h]{|ll|c|}')
            print(f'\\caption[Resultados de la métrica ``{metric}\'\' al entrenar los modelos en el conjunto de datos de {data_source} {run_modality}]{{Resultados de la métrica ``{metric}\'\' al entrenar los modelos en el conjunto de datos de {data_source} {run_modality}}}')
            print(f'\\label{{{label}}}\\\\')
            print(r'\hline')
            print(r'\multicolumn{3}{| c |}{Inicio de la tabla}\\')
            print(r'\hline')
            print(f'Modelo & Método de repr. & {metric}\\\\')
            print(r'\hline')
            print(r'\endfirsthead')

            print(r'\hline')
            print(f'\\multicolumn{{3}}{{|c|}}{{Continuación de la tabla~\\ref{{{label}}}}}\\\\')
            print(r'\hline')
            print(f'Modelo & Método de repr. & {metric}\\\\')
            print(r'\hline')
            print(r'\endhead')

            print(r'\hline')
            print(r'\endfoot')

            print(r'\hline')
            print(f'\\multicolumn{{3}}{{| c |}}{{Fin de la tabla~\\ref{{{label}}}}}\\\\')
            print(r'\hline\hline')
            print(r'\endlastfoot')

            for model in results[run_modality]:
                for embedder in results[run_modality][model]:
                    print(f'{model} & {embedder} & {results[run_modality][model][embedder][data_source][metric]} \\\\')
            print(r'\hline')
            print(r'\end{longtable}')
            print()

    metrics = ['F1 Micro', 'F1 Macro', 'F1 Ponderado', 'Pérdida de hamming']
    datasets = ['prueba', 'entrenamiento']
    for metric, dataset in product(metrics, datasets):
        make_table_for_metric_appendix(metric, dataset, ('apx:tbl:' + metric + dataset).replace(' ', '-').lower())


    def make_table_for_results_section_condensed(metric, data_source, label):
        for run_modality in results:
            print(r'\begin{table}[h]')
            print(r'\centering')
            print(f'\\caption[Resultados condensados de la métrica ``{metric}\'\' al entrenar los modelos en el conjunto de datos de {data_source} {run_modality}]{{Resultados condensados de la métrica ``{metric}\'\' al entrenar los modelos en el conjunto de datos de {data_source} {run_modality}}}')
            print(f'\\label{{{label}}}')

            print(r'\begin{tabular}{|ll|c|}')
            print(r'\hline')
            print(f'Modelo & Método de repr. & {metric}\\\\')
            print(r'\hline')
            for model in results[run_modality]:
                results_model = []
                for embedder in results[run_modality][model]:
                    results_model.append({
                        'name': embedder,
                        'value': results[run_modality][model][embedder][data_source][metric]
                    })
                best_embedder = max(results_model, key=lambda x: x['value'])
                print(f'{model} & {best_embedder["name"]} & {best_embedder["value"]} \\\\')
            print(r'\hline')
            print(r'\end{tabular}')
            print(r'\end{table}')
            print()

    print('BEGIN TABLES FOR RESULTS SECTION')
    for metric, dataset in product(metrics, datasets):
        make_table_for_results_section_condensed(metric, dataset, ('apx:tbl:condensed-' + metric + "-" + dataset).replace(' ', '-').lower())


if __name__ == '__main__':
    main()
