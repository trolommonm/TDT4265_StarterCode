import matplotlib.pyplot as plt

def plot(list_of_logs, labels, key):
    fig, ax = plt.subplots()

    for idx, logs in enumerate(list_of_logs):
        entries = [entry for entry in logs if key in entry]
        keys = [entry[key] for entry in entries]
        global_steps = [entry["global_step"] for entry in entries]

        ax.plot(global_steps, keys, label=labels[idx])

    ax.set_xlabel("Global step")
    ax.set_ylabel(key)

    return fig, ax