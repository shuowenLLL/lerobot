dataset_no_delta = LeRobotDataset(DATASET_ID)

for idx in [SAMPLE_INDEX, SAMPLE_INDEX + 1, SAMPLE_INDEX + 2, SAMPLE_INDEX + 5]:
    s = dataset_no_delta[idx]
    state = s["observation.state"].detach().cpu().numpy()
    action = s["action"].detach().cpu().numpy()

    print(f"\nidx={idx}")
    print("state =", state)
    print("action=", action)
    print("action - state =", action - state)