import json


# with open('train.txt', 'r') as file:
#     outarr = []
#     surprise = 0
#     love = 0
#     joy = 0
#     fear = 0
#     anger = 0
#     sadness = 0
#     for sent in file.readlines():
#         out = sent.split(';')
#         if out[1].startswith('surprise'):
#             surprise += 1
#         elif out[1].startswith('love'):
#             love += 1
#         elif out[1].startswith('joy'):
#             joy += 1
#         elif out[1].startswith('fear'):
#             fear += 1
#         elif out[1].startswith('anger'):
#             anger += 1
#         elif out[1].startswith('sadness'):
#             sadness += 1
#         outarr.append(out)
#     valarr = [surprise, love, joy, fear, anger, sadness]

#     with open('reduced.txt', 'w') as outfile:
#         i = 0
#         while i < len(valarr):
#             for j in range(valarr[i]//2):
#                 outfile.write(outarr[j][0] + ';' + outarr[j][1])
#             i += 1



with open('reduced.txt', 'r') as file:
    outarr = []
    surprise = 0
    love = 0
    joy = 0
    fear = 0
    anger = 0
    sadness = 0
    for sent in file.readlines():
        out = sent.split(';')
        if out[1].startswith('surprise'):
            surprise += 1
        elif out[1].startswith('love'):
            love += 1
        elif out[1].startswith('joy'):
            joy += 1
        elif out[1].startswith('fear'):
            fear += 1
        elif out[1].startswith('anger'):
            anger += 1
        elif out[1].startswith('sadness'):
            sadness += 1
        outarr.append(out)
    valarr = [surprise, love, joy, fear, anger, sadness]

print(valarr)
