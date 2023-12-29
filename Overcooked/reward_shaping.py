class detail_feature:
    def __init__(self,obs):

        self.p_i_orientation = obs[:4] ### orientation
        self.p_i_obj = obs[4:8] ## onion, soup, dish, tomato

        self.p_i_closest_onion = obs[8:10]
        # self.p_i_closest_tomato = obs[10:12]
        self.p_i_closest_dish = obs[12:14]
        self.p_i_closest_soup = obs[14:16]

        self.p_i_closest_soup_n_ingredients = obs[16:18]
        self.p_i_closest_serving_area = obs[18:20]
        self.p_i_closest_empty_counter = obs[20:22]

        ## pot
        self.p_i_closest_potj_exists = obs[22]
        self.p_i_closest_potj_status = obs[23:27] # is empty| is full| is cooking| is ready
        self.p_i_closest_potj_num = obs[27:29] # union_num | tomato_num
        self.p_i_closest_potj_cook_time = obs[29] # Remaining cooking time
        self.p_i_closest_potj = obs[30:32] # The relative position of the nearest pot

        # p_i_closest_potj_exists = obs[32]
        # p_i_closest_potj_status = obs[33:37]  # is empty| is full| is cooking| is ready
        # p_i_closest_potj_num = obs[37:39]  # union_num | tomato_num
        # p_i_closest_potj_cook_time = obs[39]  # Remaining cooking time
        # p_i_closest_potj = obs[40:42]  # The relative position of the nearest pot

        self.p_i_wall = obs[42:]  # boolean value of whether player i has a wall immediately in direction j



def calculate_rewards(events, obs, obs_pre):

    # Initialize rewards for each agent
    rewards = [0, 0]
    stay_time_union = 0
    stay_time_pot = 0
    for agent_id in range(2):
        df = detail_feature(obs[agent_id][:46])
        df_pre = detail_feature(obs_pre[agent_id][:46])
        loc = obs[agent_id][-2:]
        pre_loc = obs_pre[agent_id][-2:]
        if events[agent_id]:
            ## take union
            if df.p_i_closest_potj_num[0] < 3 and judge_str_in(events[agent_id], 'useful_onion_pickup'):
                rewards[agent_id] += 2
            ## put union
            elif df.p_i_closest_potj_num[0] < 3 and judge_str_in(events[agent_id], 'viable_onion_potting'):
                rewards[agent_id] += 5
            # take dish
            elif df.p_i_closest_potj_num[0] == 3 and judge_str_in(events[agent_id], 'dish_pickup'):
                rewards[agent_id] += 10
            # take soup
            elif df.p_i_closest_potj_status[3] and judge_str_in(events[agent_id], 'soup_pickup'):
                rewards[agent_id] += 10
            #  soup delivery
            elif judge_str_in(events[agent_id], 'soup_delivery'):
                rewards[agent_id] += 20
            # others
            else:
                rewards[agent_id] -= 2
        else:
            if pre_loc[0] == loc[0] and pre_loc[1] == loc[1]:
                rewards[agent_id] -= 1
            # with soup
            elif df.p_i_obj[1]:
                soup_dis_r = np.sqrt(df_pre.p_i_closest_serving_area[0] ** 2 + df_pre.p_i_closest_serving_area[1] ** 2) - \
                           np.sqrt(df.p_i_closest_serving_area[0] ** 2 + df.p_i_closest_serving_area[1] ** 2)
                rewards[agent_id] += soup_dis_r
            elif df.p_i_closest_potj_num[0] < 3:
                # hold union
                if df.p_i_obj[0]:
                    pot_dis_r = np.sqrt(df_pre.p_i_closest_potj[0] ** 2 + df_pre.p_i_closest_potj[1] ** 2) - \
                                np.sqrt(df.p_i_closest_potj[0] ** 2 + df.p_i_closest_potj[1] ** 2)
                    rewards[agent_id] += pot_dis_r
                # without
                else:
                    onion_dis_r = np.sqrt(df_pre.p_i_closest_onion[0] ** 2 + df_pre.p_i_closest_onion[1] ** 2) - \
                                  np.sqrt(df.p_i_closest_onion[0] ** 2 + df.p_i_closest_onion[1] ** 2)
                    if onion_dis_r == 0:
                        rewards[agent_id] -= 1
                    rewards[agent_id] += onion_dis_r
                    # ### stay
                    # if np.sqrt(df.p_i_closest_onion[0] ** 2 + df.p_i_closest_onion[1] ** 2) == np.sqrt(df_pre.p_i_closest_onion[0] ** 2 + df_pre.p_i_closest_onion[1] ** 2):
                    #     stay_time_union += 1
                    #     rewards[agent_id] -= 1

            elif df.p_i_closest_potj_num[0] == 3:
                # with dish
                if df.p_i_obj[2]:
                    dish_2_pot_dis_r = np.sqrt(df_pre.p_i_closest_potj[0] ** 2 + df_pre.p_i_closest_potj[1] ** 2) - \
                                       np.sqrt(df.p_i_closest_potj[0] ** 2 + df.p_i_closest_potj[1] ** 2)
                    rewards[agent_id] += dish_2_pot_dis_r
                else:
                    dish_dis_r = np.sqrt(df_pre.p_i_closest_dish[0] ** 2 + df_pre.p_i_closest_dish[1] ** 2) - \
                                 np.sqrt(df.p_i_closest_dish[0] ** 2 + df.p_i_closest_dish[1] ** 2)

                    if dish_dis_r == 0:
                        rewards[agent_id] -= 1
                    else:
                        rewards[agent_id] += dish_dis_r

                    ## stay
                    if np.sqrt(df.p_i_closest_potj[0] ** 2 + df.p_i_closest_potj[1] ** 2) == np.sqrt(
                            df_pre.p_i_closest_potj[0] ** 2 + df_pre.p_i_closest_potj[1] ** 2):
                        # stay_time_pot += 1
                        rewards[agent_id] -= 1

            # elif
    return rewards