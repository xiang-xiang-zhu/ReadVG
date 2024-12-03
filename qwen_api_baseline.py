# coding:utf-8

# model qwen-max
import time
from http import HTTPStatus
import dashscope
from tqdm import tqdm
import re
import json

api_key = ''
dashscope.api_key=api_key


prompt = '''作为一个生成视觉描述的专家，你的任务是根据一段text和一个Query，给出Query中目标的视觉描述。描述的重点是目标本身，包括发色、性别、衣服以及任何独特的配饰（眼镜或大型首饰）。一定不要提及任何人名（用man、woman、boy、girl代替）、吸引力、眼睛颜色、身体尺寸、细小的视觉细节、特定的服装品牌，除非它们与众不同。视觉描述一定不能超过8个单词，也不能出现任何标点符号。你必须用英文输出。
下面是一个示例，你的输出应该遵循示例的格式：
示例："Text:David is standing behind the billiards table, smoking a cigarette and taking off his clothes. He wants to play billiards on the billiards table in front of him. The man in black behind David is his brother Eric. He is very tired and leaving for home. David's friend Peter is sitting on the chair behind David's left. He is wearing a white coat and looks at David.
Query:The man who is taking off his clothes
the smoking man"
输入："Text:{}
Query:{}"'''


def call_with_prompt(prompt):
    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_turbo,
        prompt=prompt
    )
    # 如果调用成功，则打印模型的输出
    if response.status_code == HTTPStatus.OK:
        return response['output']['text']
    # 如果调用失败，则打印出错误码与失败信息
    else:
        print(response.code)
        print(response.message)

def extract_name(s):
    match = re.search(r'Person\s*(?:,|:)\s*(\w+)', s)
    if match:
        return match.group(1)
    else:
        return None

if __name__ == '__main__':
    sks = []
    f = open('./sk-vg.v1/annotations.json', 'r', encoding='utf-8')
    anno_data = json.load(f)
    test_data = anno_data['test']

    res = []
    cnt = 1
    for i in tqdm(range(int((cnt-0.1)*len(test_data)), int(cnt * len(test_data)))):
        tmp_dict = {}
        res_1 = call_with_prompt(prompt=prompt.replace('{}', test_data[i]['knowledge'], 1).replace('{}', test_data[i]['ref_exp'], 1))
        # time.sleep(1.5)
        tqdm.write('res 1: {}'.format(res_1))
        tmp_dict['query'] = res_1

        tmp_dict['level'] = test_data[i]['level']
        res.append(tmp_dict)

    # json_string = json.dumps(res, indent=8)
    with open('ours_{}_baseline.json'.format(str(int(cnt*10))), 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)