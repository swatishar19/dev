import pandas as pd
from scipy.stats import fisher_exact
from collections import defaultdict
import time

toxprint = pd.read_csv('toxprintV2_vs_curatedDev_allspecies shengde,fda,caeser,toxref.csv')
toxprint= toxprint.set_index('ID')

bioprofile = pd.read_csv('filtered_trainingset_DEV_shengdeFDAtoxCAESER_bioprofile_woRemovingCorrelatedAssays.csv')
bioprofile = bioprofile.set_index('ID')
tox = toxprint.loc[bioprofile.index]

# print(tox)

total_counts = bioprofile.shape[1]

counter1 = 0

aid_fp_dict = defaultdict(dict)
aid_fp_dict_raw = defaultdict(dict)

for aid, assay_data in bioprofile.iteritems():
    assay_data_wo_null = assay_data.dropna()
    actives = assay_data_wo_null[assay_data_wo_null == 1]
    inactives = assay_data_wo_null[assay_data_wo_null == -1]
    # print("The number of compounds in AID {aid} is {cmps}.".format(aid=aid,
    #                                                               cmps=assay_data_wo_null.shape[0]))
    # print("The number of active respones is {no_act}".format(no_act=actives.shape[0]))
    # print("The number of inactive respons is {no_inact}".format(no_inact=inactives.shape[0]))



    # only check aid if actives and inactives > 0
    if (actives.shape[0] and inactives.shape[0]):
        counter2 = 0
        for fp, fp_data in tox.iteritems():
            cmps_with_fp = fp_data[fp_data == 1]
            cmps_without_fp = fp_data[fp_data == 0]

            # find the distributions of active/inactive responses
            # and absence/presence of fp
            act_and_fp = actives.index.intersection(cmps_with_fp.index).shape[0]
            act_and_wo_fp = actives.index.intersection(cmps_without_fp.index).shape[0]

            inact_and_fp = inactives.index.intersection(cmps_with_fp.index).shape[0]
            inactive_and_wo_fp = inactives.index.intersection(cmps_without_fp.index).shape[0]


            assert (actives.shape[0] + inactives.shape[0]) == (act_and_fp + act_and_wo_fp + \
                                                                        inact_and_fp + inactive_and_wo_fp)

            confusion_matrix = [
                                    [act_and_fp, act_and_wo_fp],
                                    [inact_and_fp, inactive_and_wo_fp]
            ]

            oddsratio, pvalue = fisher_exact(confusion_matrix)

            if pvalue < 0.05:
                # print("Actives with fp: {0}, Actives without fp: {1}".format(act_and_fp, act_and_wo_fp))
                # print("Inactives with fp: {0}, Inactives without fp: {1}".format(inact_and_fp, inactive_and_wo_fp))
                # print()
                # print("The p-value for AID {0} and FP {1} is {2}".format(aid, fp, pvalue))

                # make a connection in the aid - fingerprint graph
                aid_fp_dict[aid][fp] = 1
            aid_fp_dict_raw[aid][fp] = pvalue
            #     counter2 = counter2 + 1
            if counter2 > 10:
                break
    counter1 = counter1 + 1
    # if counter1 > 60:
    #     break
    print("{0:.2f}% completed".format(counter1/total_counts * 100))

pd.DataFrame.from_dict(aid_fp_dict, orient='index').fillna(0).to_csv('aidXfpP-valuebinary.csv')
# pd.DataFrame.from_dict(aid_fp_dict_raw, orient='index').to_csv(PROJECT_DIR + '/resources/txt/aidXfp2_raw.csv')
print(pd.DataFrame.from_dict(aid_fp_dict, orient='index').fillna(0))
print("{0:.2f} s to complete".format(time.time() - start_time))