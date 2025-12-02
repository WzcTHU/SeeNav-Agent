import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import numpy as np
from time import sleep
from envs.EBNavEnv import EBNavigationEnv, ValidEvalSets
from tqdm import tqdm
from envs.utils.EBNav_env_utils import average_json_values, logger
from planner.utils.EBNav_planner_utils import eb_navigation_system_prompt
from planner.EBNavPlanner import EBNavigationPlanner
from ActionMarker.ActionMarker import ActionMarker


class EB_NavigationEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_sets = config["eval_sets"]
        self.eval_set = None
        self.config = config

        self.env = None
        self.planner = None

    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)
    
    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        self.eval_sets = list(valid_eval_sets)
        if type(self.eval_sets) == list and len(self.eval_sets) == 0:
            self.eval_sets = ValidEvalSets
            
        for eval_set in self.eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"

            self.env = EBNavigationEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], 
                                    exp_name=exp_name, multiview=self.config['multiview'], boundingbox=self.config['detection_box'], 
                                    multistep = self.config['multistep'], resolution = self.config['resolution'], 
                                    target_only=self.config['target_only'], log_path=self.config['log_path'], full_config=self.config)

            self.planner = EBNavigationPlanner(model_name=self.model_name, model_type = self.config['model_type'], 
                                            actions=self.env.language_skill_set, system_prompt=eb_navigation_system_prompt, 
                                            examples=[], n_shot=self.config['n_shots'], obs_key='auto', 
                                            chat_history=self.config['chat_history'], language_only=self.config['language_only'], 
                                            multiview=self.config['multiview'], multistep=self.config['multistep'], 
                                            visual_icl=self.config['visual_icl'], truncate=self.config.get('truncate', False), config=self.config)
            
            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), selected_key = None)
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': []}
            obs = self.env.reset()
            new_obs = self.env.add_marks_on_obs(obs, fv_bbox=self.config['fv_bbox'], bev_bbox=self.config['bev_bbox'])
            if self.config['save_image'] and not self.config['image_concat']:
                    self.env.save_image(new_obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")
            self.planner.reset()
            done = False
            while not done:
                self.ActionMarker = ActionMarker(env=self.env.env, env_name='EB-Navigation', config={'img_height': self.config['resolution'], 'img_width': self.config['resolution']})
                self.ActionMarker.episode_idx = self.env._current_episode_num
                self.ActionMarker.planner_steps = self.env._current_step

                action, reasoning = self.planner.act(new_obs, user_instruction, projector=self.ActionMarker)
                print(f"Planner Output Action: {action}")
                reasoning = json.loads(reasoning)
                if type(action) == list:
                    for i, action_single in enumerate( action[:min(self.env._max_episode_steps - self.env._current_step + 1, len(action))] ):
                        last_dist = self.env.measure_success()
                        last_visible = self.env.check_target_visibility()
                        if i==0:
                            obs, reward, done, info = self.env.step(action_single,reasoning,1)
                        else:
                            obs, reward, done, info = self.env.step(action_single,reasoning,0)

                        if self.config['record_sft_sample']:
                            # 如果离目标更近，并且动作valid，就记录当前的状态-动作
                            # 如果能把目标从看不见变成看见，就记录当前的状态-动作
                            now_dist = self.env.measure_success()
                            now_visible = self.env.check_target_visibility()
                            if (now_dist < last_dist and info['last_action_success'] == 1) or (now_visible and not last_visible):
                                sft_sample = self.planner.sft_sample
                                sft_sample_to_save = {
                                    "conversations": sft_sample['conversations'],
                                    "images": [sft_sample['image_path']]
                                }
                                image_to_save = sft_sample['image']
                                image_to_save.save(sft_sample['image_path'])
                                with open(os.path.join(self.config['sft_sample_path'], 'sft_dataset.jsonl'), 'a') as f:
                                    json_str = json.dumps(sft_sample_to_save, ensure_ascii=False)
                                    f.write(json_str + '\n')
                            else:
                                pass


                        print(f"Executed action: {action_single}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        self.planner.update_info(info)
                        new_obs = self.env.add_marks_on_obs(obs, fv_bbox=self.config['fv_bbox'], bev_bbox=self.config['bev_bbox'])
                        if self.config['save_image']:
                            if not self.config['image_concat']:
                                self.env.save_image(new_obs)

                        episode_info['reward'].append(reward)

                        if done==True:
                            break

                        if info['last_action_success'] == 0:
                            # stop for replanning
                            print('invalid action, start replanning')
                            break
                else:
                    obs, reward, done, info = self.env.step(action, reasoning, 1)
                    print(f"Executed action: {action}, Task success: {info['task_success']}")
                    logger.debug(f"reward: {reward}")
                    logger.debug(f"terminate: {done}\n")
                    self.planner.update_info(info)
                    new_obs = self.env.add_marks_on_obs(obs, fv_bbox=self.config['fv_bbox'], bev_bbox=self.config['bev_bbox'])
                    if self.config['save_image']:
                        self.env.save_image(new_obs)
                    episode_info['reward'].append(reward)

                # except Exception as e:
                #     sleep(1)
                #     print(e)
                #     print("retrying...")

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            # episode_info["task_progress"] = info['task_progress']
            # episode_info['subgoal_reward'] = info['subgoal_reward']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            # episode_info["num_invalid_actions"] = info["num_invalid_actions"]
            # episode_info["num_invalid_action_ratio"] = info["num_invalid_actions"] / info["env_step"]
            episode_info["episode_elapsed_seconds"] = info["episode_elapsed_seconds"]
            self.save_episode_metric(episode_info)
            progress_bar.update()

if __name__ == "__main__":
    config = {
        # 'model_name': 'GPT4.1',
        'model_name': 'Qwen2.5-VL-3B-Instruct',
        'down_sample_ratio': 1, # 0.000666,
        'model_type': 'local',
        'local_model_path': 'your_model_path',
        'language_only': False,
        # 'dataset': sys.argv[1],
        'eval_sets': ['base'],
        # 'eval_sets': ['ICL'],
        'chat_history': True, 
        'action_num_per_plan': 5,
        'fov': 100,
        'n_shots' : 0,   # int(sys.argv[3])
        'sleep_time':  0, #int(sys.argv[3]),
        # 'boundingbox': 0, #sys.argv[4]=='1',
        # 'target_only': 0, #sys.argv[5]=='1',
        'multistep':0, #sys.argv[6]=='1',
        'resolution': 500, #int(sys.argv[7]),
        'purpose': "retest", #sys.argv[8],
        'exp_name': 'SFT-Qwen-Nav',
        'icl_abl':0, #sys.argv[10]=='1',
        'visual':0, #sys.argv[11]=='1',
        'target_only': True,
        'log_path': 'your_log_path',
        'save_image': True,
        'visual_icl': False,
        'topK_action': 8,
        'detection_box': True,

        'planning': False,
        'multiview': True,
        'action_project': True,
        'image_concat': True,
        'nav_line': True,
        'align': True,
        'agent_mark': True,
        'fv_bbox': True,
        'bev_bbox': True,

        'record_sft_sample': False,
        'sft_sample_path': 'your_save_path',
        'random_reset': False
    }
    if not os.path.exists(config['log_path']):
        os.makedirs(config['log_path'])
    if config['record_sft_sample'] and not os.path.exists(config['sft_sample_path']):
        os.makedirs(config['sft_sample_path'])
    evaluator = EB_NavigationEvaluator(config)
    evaluator.evaluate_main()