# pip install zhipuai 请先在终端进行安装

from zhipuai import ZhipuAI
import zhipuai
import json
import re
from tqdm import tqdm
import time

client = ZhipuAI(api_key="")

prompt_1 = '''给定一段text和一个query，你的回答是query中目标的种类和名称，种类包含Person和Item。
下面是几个例子，你的输出应该遵循示例的格式，只用输出目标的名字就可以了：
示例1："Text:John, who is a boy in a hat, is drinking with his friends. His girlfriend Sarah is on his right. John's friend Peter is on John's left, holding a glass and toasting his friends. Rick is sitting between John and Peter, laughing happily.
Query:The hat worn by John
Item:hat"
示例2："text:David is standing behind the billiards table, smoking a cigarette and taking off his clothes. He wants to play billiards on the billiards table in front of him. The man in black behind David is his brother Eric. He is very tired and leaving for home. David's friend Peter is sitting on the chair behind David's left. He is wearing a white coat and looks at David.
query:The man who is taking off his clothes
Person:David"
输入："Text:{}
Query:{}"'''

prompt_2 = '''作为一个生成视觉描述的专家，你的任务是根据一段text和一个目标的名称，给出text中目标的视觉描述。描述的重点是目标本身，包括发色、性别、衣服以及任何独特的配饰（眼镜或大型首饰）。一定不要提及任何人名（用man、woman、boy、girl代替）、吸引力、眼睛颜色、身体尺寸、细小的视觉细节、特定的服装品牌，除非它们与众不同。视觉描述一定不能超过8个单词，也不能出现任何标点符号。
下面是几个示例，你的输出应该遵循示例的格式：
示例1："Text:David is standing behind the billiards table, smoking a cigarette and taking off his clothes. He wants to play billiards on the billiards table in front of him. The man in black behind David is his brother Eric. He is very tired and leaving for home. David's friend Peter is sitting on the chair behind David's left. He is wearing a white coat and looks at David.
Name:David
the smoking man"
示例2：""
输入："Text:{}
Name:{}"'''

def call_glm(prompt):
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        top_p= 0.7,
        temperature= 0.95,
        max_tokens=1024,
        tools = [{"type":"web_search","web_search":{"search_result":True}}],
        stream=False
    )

    return response

def extract_name(s):
    match = re.search(r'Person\s*(?:,|:)\s*(\w+)', s)
    if match:
        return match.group(1)
    else:
        return None

# for trunk in response:
#     print(trunk)
#
# print(response.choices[0].message.content)

sks = []
f = open('./sk-vg.v1/annotations.json', 'r', encoding='utf-8')
anno_data = json.load(f)
test_data = anno_data['test']

res = []
cnt = 1
for i in tqdm(range(int((cnt-0.1)*len(test_data)), int(cnt * len(test_data)))):
    tmp_dict = {}
    res_1 = call_glm(prompt=prompt_1.replace('{}', test_data[i]['knowledge'], 1).replace('{}', test_data[i]['ref_exp'], 1))
    # time.sleep(1.5)
    res_1 = res_1.choices[0].message.content
    tqdm.write('res 1: {}'.format(res_1))
    tmp_dict['name'] = extract_name(res_1)

    if 'Person' in res_1:
        try:
            res_2 = call_glm(
                prompt=prompt_2.replace('{}', test_data[i]['knowledge'], 1).replace('{}', tmp_dict['name'], 1))
            res_2 = res_2.choices[0].message.content
            time.sleep(1.5)
        except zhipuai.core._errors.APIRequestFailedError as e:
            res_2 = ''
            tqdm.write(f"Error calling GLM for index {i}: {e}")
    elif 'Item' in res_1:
        res_2 = res_1.split(':')[1]
    else:
        res_2 = res_1

    tqdm.write('res 2: {}'.format(res_2))
    tmp_dict['query'] = res_2
    tmp_dict['level'] = test_data[i]['level']
    res.append(tmp_dict)

with open('ours_{}_glm.json'.format(str(int(cnt*10))), 'w') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)