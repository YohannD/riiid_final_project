import pandas as pd
import numpy as np
import random
import math as m

def user_history_update(content_type_id,
                        content_id,
                        qstats,
                        user_history=None,
                        mode='training',    #autre choix : mode exam
                        prior_question_had_explanation=False):

    '''Crée ou met à jour l'hisorique d'un utilisateur, stockée dans un df'''

    if not type(user_history)==pd.DataFrame:
        user_history=pd.DataFrame({
                             #following columns are the impute of each loop
                             ### TO BE IMPUTED ###
                             'content_id':[-1],
                             'content_type_id':[-1],
                             'prior_question_had_explanation':False,
                             'mode':'n/a',
                             # following columns depend of previous history of the user :
                             ### TO BE UPDATED WHATEVER THE CONTENT_TYPE ###
                             'user_activity_cumcount':[-1],
                             ### TO BE UPDATED IF LAST WAS LECTURE ###
                             'at_least_one_lesson':[0],
                             ### TO BE UPDATED IF LAST WAS QUESTION ###
                             'user_avg_score_cum':[0.499],
                             'user_correct_answers_cum':[0],
                             'user_avg_score_cum_part1':[0.499],
                             'user_avg_score_cum_part2':[0.499],
                             'user_avg_score_cum_part3':[0.499],
                             'user_avg_score_cum_part4':[0.499],
                             'user_avg_score_cum_part5':[0.499],
                             'user_avg_score_cum_part6':[0.499],
                             'user_avg_score_cum_part7':[0.499],
                             'user_correct_answers_cum_part1':[0],
                             'user_correct_answers_cum_part2':[0],
                             'user_correct_answers_cum_part3':[0],
                             'user_correct_answers_cum_part4':[0],
                             'user_correct_answers_cum_part5':[0],
                             'user_correct_answers_cum_part6':[0],
                             'user_correct_answers_cum_part7':[0],
                             # following columns are pure question stats :
                             ### TO BE IMPORTED FROM QUESTIONS ###
                             'part':[-1],
                             'qstats_answered_correctly':[-1],
                             'qstats_prior_question_had_explanation':[-1],
                             'qstats_answered_correctly_knowing_having_had_explanation':[-1],
                             'qstats_answered_correctly_knowing_having_not_had_explanation':[-1],
                             # following columns depend of the current question AND the hisory of user
                             ### TO BE COMPUTED ###
                             'user_personalized_qstat_knowing_had_explanation_or_not':[-1],
                             'already_seen':[-1],
                             'user_avg_score_cum_on_this_part':[-1],
                             'user_correct_answers_cum_on_this_part':[-1],
                             # the following line is the prediction to be made
                             ### TO BE PREDICTED ###
                             'answered_correctly':[-1]
                          })


    last_line=user_history.iloc[-1]
    new_line =last_line.copy()

    last_content_type_id=user_history.iloc[-1]['content_type_id']

    ### TO BE IMPUTED ###
    new_line['content_id']=content_id
    new_line['content_type_id']=content_type_id
    new_line['prior_question_had_explanation']=prior_question_had_explanation
    new_line['mode']=mode
    ### TO BE UPDATED WHATEVER THE CONTENT_TYPE ###
    new_line['user_activity_cumcount'] += 1

    if mode=='training':
        if last_content_type_id==0:
            part=last_line['part']
            ### TO BE UPDATED IF LAST WAS QUESTION ###
            new_line['user_correct_answers_cum'] = last_line['user_correct_answers_cum']\
                                                 + last_line['answered_correctly']
            new_user_questions_count             = last_line['user_correct_answers_cum']\
                                                 / (last_line['user_avg_score_cum']+0.000001)\
                                                 + 1
            new_line['user_avg_score_cum']       = new_line['user_correct_answers_cum']\
                                                 / new_user_questions_count

            new_line[f'user_correct_answers_cum_part{part}'] = last_line[f'user_correct_answers_cum_part{part}']\
                                                             + last_line['answered_correctly']
            vars()[f'new_user_questions_count_part{part}']   = last_line[f'user_correct_answers_cum_part{part}']\
                                                             / (last_line[f'user_avg_score_cum_part{part}']+0.000001)\
                                                             + 1
            new_line[f'user_avg_score_cum_part{part}']       = new_line[f'user_correct_answers_cum_part{part}']\
                                                             / (vars()[f'new_user_questions_count_part{part}']+0.000001)

        elif last_content_type_id==1:
            ### TO BE UPDATED IF LAST WAS LECTURE ###
            new_line['at_least_one_lesson']=1

    if content_type_id==0:
        currect_question_stats=qstats.loc[qstats.content_id==content_id].iloc[-1]
        ### TO BE IMPORTED FROM QUESTIONS ###
        new_line['part']\
              = currect_question_stats['part']
        new_line['qstats_answered_correctly']\
              = currect_question_stats['qstats_answered_correctly']
        new_line['qstats_prior_question_had_explanation']\
              = currect_question_stats['qstats_prior_question_had_explanation']
        new_line['qstats_answered_correctly_knowing_having_had_explanation']\
              = currect_question_stats['qstats_answered_correctly_knowing_having_had_explanation']
        new_line['qstats_answered_correctly_knowing_having_not_had_explanation']\
              = currect_question_stats['qstats_answered_correctly_knowing_having_not_had_explanation']
        ### TO BE COMPUTED ###
        new_line['user_personalized_qstat_knowing_had_explanation_or_not']\
              = new_line['qstats_answered_correctly_knowing_having_had_explanation']\
             if prior_question_had_explanation\
           else new_line['qstats_answered_correctly_knowing_having_not_had_explanation']
        new_line['already_seen']\
              = 1 if content_id in user_history.loc[user_history.content_type_id==0,'content_id']\
           else 0
        new_line['user_avg_score_cum_on_this_part']=new_line[f'user_avg_score_cum_part{new_line["part"]}']
        new_line['user_correct_answers_cum_on_this_part']=new_line[f'user_correct_answers_cum_part{new_line["part"]}']

    elif content_type_id==1:

        ### TO BE IMPORTED ###
        new_line['part']= -1 # TODO : si on veut utiliser la partie de la lecture, il faut importer la base des lectures
        new_line['qstats_answered_correctly']= -1
        new_line['qstats_prior_question_had_explanation']= -1
        new_line['qstats_answered_correctly_knowing_having_had_explanation']= -1
        new_line['qstats_answered_correctly_knowing_having_not_had_explanation']= -1
        ### TO BE COMPUTED ###
        new_line['user_personalized_qstat_knowing_had_explanation_or_not']= -1
        new_line['already_seen']= -1
        new_line['user_avg_score_cum_on_this_part']= -1
        new_line['user_correct_answers_cum_on_this_part']= -1

    ### TO BE PREDICTED ###
    new_line['answered_correctly']= -1

    user_history=user_history.append(new_line,ignore_index=True)

    return user_history


def pick_a_question(qstats,
                    qselection_by_part_and_level,
                    user_history_last_line,
                    strategy='random'):
    import math as m
    if strategy=='random':
        question_id=random.choice(qstats.content_id.to_list())

    elif strategy=='knowledge_tracing':
        competences=[]
        for i in range(7):
            competences.append(user_history_last_line[f'user_avg_score_cum_part{i+1}']\
                              *m.sqrt(user_history_last_line[f'user_correct_answers_cum_part{i+1}']))
        weakest_part=np.argmin(competences)+1
        weakest_value=min(competences)
        weakest_level=0
        if weakest_value > 0.6:
            weakest_level=1
        if weakest_value > 2.5:
            weakest_level=2
        if weakest_value > 12:
            weakest_level=3

        question_id=random.choice(qselection_by_part_and_level[weakest_part-1,weakest_level])

    else:
        question_id=-1
    return question_id

def training (my_pipeline,
              pipeline_features_list,
              qstats,
              qselection_by_part_and_level=None,
              user_history=None,
              loop_length=30,
              question_selection_strategy='random'):
    '''only the random strategy implemented yet'''
    for i in range(loop_length):
        ### CHOIX DE LA QUESTION ###
        if question_selection_strategy=='random':
            next_question_id=random.choice(qstats.content_id.to_list())


            user_history=user_history_update(0,
                                         next_question_id,
                                         qstats,
                                         qselection_by_part_and_level,
                                         user_history,
                                         mode='training',
                                         prior_question_had_explanation=random.uniform(0, 1)>0.1)

        ### PREDICTION ###
            user_history.iloc[-1,-1]\
                = my_pipeline.predict_proba(user_history[pipeline_features_list.feature.to_list()].iloc[-2:-1])[0,1]
    return user_history



def TOEIC_scoring (my_pipeline,
                   pipeline_features_list,
                   qstats,
                    user_history,
                   number_of_questions=100,
                   TOEIC_strategy='random'):
    '''only the random strategy implemented yet'''

    for i in range(number_of_questions):
        ### CHOIX DE LA QUESTION ###
        if TOEIC_strategy=='random':
            next_question_id=random.choice(qstats.content_id.to_list())


        user_history=user_history_update(0,
                                         next_question_id,
                                         qstats,
                                         user_history,
                                         mode='exam',
                                         prior_question_had_explanation=False)

        ### PREDICTION ###
    user_history.iloc[-number_of_questions:,-1]\
            = my_pipeline.predict_proba(user_history[pipeline_features_list.feature.to_list()].iloc[-number_of_questions:])[:,1]

    return user_history.iloc[-number_of_questions:].answered_correctly.mean()


def initialize_profile(experience_list):
    '''Takes a list of 7 values, one per part
            0=full beginner,
            1=intermediate,
            2=average,
            3=fluent
            exemple : [0,1,1,3,1,2,0]

        Returns a one-line user_history
        with the average and cumulated score per part initialized
            '''
    parts_mean_perfs=[0.7394,
                      0.7107,
                      0.6947,
                      0.6184,
                      0.6121,
                      0.6767,
                      0.6657]
    parts_mean_correct_count=[18.8,
                              35.0,
                              27.4,
                              24.3,
                              65.2,
                              32.6,
                              16.1]
    parts_max_correct_count=[689,
                             2653,
                             1247,
                             981,
                             3561,
                             1363,
                             1012]

    user_avg_score_cum_parts=[]
    user_correct_answers_cum_parts=[]
    for i in range(7):
        if experience_list[i]==0:
            user_avg_score_cum_parts.append(0)
            user_correct_answers_cum_parts.append(0)
            at_least_one_lesson=0
        if experience_list[i]==1:
            user_avg_score_cum_parts.append(parts_mean_perfs[i]/2)
            user_correct_answers_cum_parts.append(parts_mean_correct_count[i]/2)
            at_least_one_lesson=0
        if experience_list[i]==2:
            user_avg_score_cum_parts.append(parts_mean_perfs[i])
            user_correct_answers_cum_parts.append(parts_mean_correct_count[i])
            at_least_one_lesson=1
        if experience_list[i]==3:
            user_avg_score_cum_parts.append(1)
            user_correct_answers_cum_parts.append(parts_max_correct_count[i])
            at_least_one_lesson=1

    user_correct_answers_cum=sum(user_correct_answers_cum_parts)
    user_avg_score_cum=user_correct_answers_cum\
                       /(sum([user_correct_answers_cum_parts[i]\
                              /(user_avg_score_cum_parts[i]+0.000001) for i in range(7)])+0.000001)


    user_profile=pd.DataFrame({
                             #following columns are the impute of each loop
                             ### TO BE IMPUTED ###
                             'content_id':[-1],
                             'content_type_id':[-1],
                             'prior_question_had_explanation':False,
                             'mode':'n/a',
                             # following columns depend of previous history of the user :
                             ### TO BE UPDATED WHATEVER THE CONTENT_TYPE ###
                             'user_activity_cumcount':[-1],
                             ### TO BE UPDATED IF LAST WAS LECTURE ###
                             'at_least_one_lesson':[at_least_one_lesson],
                             ### TO BE UPDATED IF LAST WAS QUESTION ###
                             'user_avg_score_cum':[user_avg_score_cum],
                             'user_correct_answers_cum':[user_correct_answers_cum],
                             'user_avg_score_cum_part1':[user_avg_score_cum_parts[0]],
                             'user_avg_score_cum_part2':[user_avg_score_cum_parts[1]],
                             'user_avg_score_cum_part3':[user_avg_score_cum_parts[2]],
                             'user_avg_score_cum_part4':[user_avg_score_cum_parts[3]],
                             'user_avg_score_cum_part5':[user_avg_score_cum_parts[4]],
                             'user_avg_score_cum_part6':[user_avg_score_cum_parts[5]],
                             'user_avg_score_cum_part7':[user_avg_score_cum_parts[6]],
                             'user_correct_answers_cum_part1':[user_correct_answers_cum_parts[0]],
                             'user_correct_answers_cum_part2':[user_correct_answers_cum_parts[1]],
                             'user_correct_answers_cum_part3':[user_correct_answers_cum_parts[2]],
                             'user_correct_answers_cum_part4':[user_correct_answers_cum_parts[3]],
                             'user_correct_answers_cum_part5':[user_correct_answers_cum_parts[4]],
                             'user_correct_answers_cum_part6':[user_correct_answers_cum_parts[5]],
                             'user_correct_answers_cum_part7':[user_correct_answers_cum_parts[6]],
                             # following columns are pure question stats :
                             ### TO BE IMPORTED FROM QUESTIONS ###
                             'part':[-1],
                             'qstats_answered_correctly':[-1],
                             'qstats_prior_question_had_explanation':[-1],
                             'qstats_answered_correctly_knowing_having_had_explanation':[-1],
                             'qstats_answered_correctly_knowing_having_not_had_explanation':[-1],
                             # following columns depend of the current question AND the hisory of user
                             ### TO BE COMPUTED ###
                             'user_personalized_qstat_knowing_had_explanation_or_not':[-1],
                             'already_seen':[-1],
                             'user_avg_score_cum_on_this_part':[-1],
                             'user_correct_answers_cum_on_this_part':[-1],
                             # the following line is the prediction to be made
                             ### TO BE PREDICTED ###
                             'answered_correctly':[-1]
                          })
    return user_profile


def plot_learning_curve(my_pipeline,
                        pipeline_features_list,
                        qstats,
                        initial_experience=[0,0,0,0,0,0,0],
                        number_students=5,
                        training_batch_size=10,
                        number_of_training_batches=10,
                        training_question_selection_strategy='random'):

    ### IF STRATEGY==knowledge_tracing, we make a selection of 7*4 lists of 50 questions for each part/level
    qselection_by_part_and_level=[]
    if training_question_selection_strategy=='knowledge_tracing':
        for i in range(7):
            questios=qstats.loc[qstats.part==i+1,['qstats_answered_correctly','content_id']].sort_values('qstats_answered_correctly')
            number_of_questions=len(questios)
            list0=questios.iloc[:50].content_id.to_list()
            tiers=int(number_of_questions/3)
            list1=questios.iloc[tiers-25:tiers+25].content_id.to_list()
            list2=questios.iloc[2*tiers-25:2*tiers+25].content_id.to_list()
            list3=questios.iloc[50:].content_id.to_list()
            qselection_by_part_and_level.append([list0,list1,list2,list3])
        qselection_by_part_and_level=np.array(qselection_by_part_and_level)

    result_moy=[]

    for i in range(number_students):
        print(f'début essai {i+1}/{number_students}')
        results=[]
        training_questions=[]

        user_history=initialize_profile(initial_experience)
        results.append(TOEIC_scoring(my_pipeline,
                                     pipeline_features_list,
                                     qstats,
                                     user_history))
        training_questions.append(0)

        for j in range(number_of_training_batches):
            training_questions.append(training_questions[-1]+training_batch_size)
            print(f'entrainement {j*training_batch_size}/{training_batch_size*number_of_training_batches}')

            user_history=training (my_pipeline,
                                   pipeline_features_list,
                                   qstats,
                                   qselection_by_part_and_level,
                                   user_history,
                                   loop_length=training_batch_size,
                                   question_selection_strategy=training_question_selection_strategy)
            results.append(TOEIC_scoring(my_pipeline,
                                         pipeline_features_list,
                                         qstats,
                                         user_history))

        result_moy.append(results)
    stats=np.array(result_moy)
    return pd.DataFrame({'training_questions':training_questions,'TOEIC_score':stats.mean(axis=0)}).set_index('training_questions')
