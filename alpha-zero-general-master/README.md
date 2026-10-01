# 香港鬥獸棋 AlphaZero agent (for suragnair/alpha-zero-general)

Copy everything in this folder into the root of the alpha-zero-general repo
(next to Coach.py, MCTS.py, main.py), and put your board photo there as board.png.

    pip install torch numpy tqdm coloredlogs pygame

Train (checkpoints go to ./temp_jungle/):

    python main_jungle.py

Test the agent:

    python pit_jungle.py --opponent greedy --games 20
    python pit_jungle.py                    # you vs the agent in the terminal

Play it on the photo board:

    python jungle_gui.py --ai top           # the agent plays 中環, you play 西九
    python jungle_gui.py --ai bottom --sims 200

Files
- jungle/JungleLogic.py   rules on a signed numpy board (default house rules)
- jungle/JungleGame.py    Game subclass; draw limits: 100 plies without a capture, 300 plies in total
- jungle/JunglePlayers.py random, greedy and human players
- jungle/pytorch/         residual network and NNetWrapper
- main_jungle.py          training settings
- pit_jungle.py           agent vs human / greedy / random
- az_bot.py               lets jungle_gui.py use a checkpoint
