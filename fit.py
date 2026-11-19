import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv("clinic_appointments.csv")
y = (df["no_show"] == "no-show").astype(int)
X = df.drop(columns=["no_show"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("X_train", X_train.shape, "no-shows", int(y_train.sum()))
print("X_test", X_test.shape, "no-shows", int(y_test.sum()))
print("y_train", y_train.shape)
print("y_test", y_test.shape)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print(export_text(model, feature_names=list(X.columns), show_weights=True))

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    model,
    filled=True,
    feature_names=list(X.columns),
    class_names=["checked-in", "no-show"],
    fontsize=9,
    ax=ax,
)
fig.savefig("tree.png")
plt.close(fig)


def value(tn, fp, fn, tp):
    return -60 * fp - 120 * fn


def report(name, model, X, y):
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(name)
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print("TP", tp)
    print("accuracy", f"{accuracy_score(y, pred):.3f}")
    print("precision", f"{precision_score(y, pred):.3f}")
    print("recall", f"{recall_score(y, pred):.3f}")


report("train", model, X_train, y_train)
report("test", model, X_test, y_test)

baseline_pred = [0] * len(y_test)
b_tn, b_fp, b_fn, b_tp = confusion_matrix(y_test, baseline_pred, labels=[0, 1]).ravel()
print("baseline")
print("TN", b_tn)
print("FP", b_fp)
print("FN", b_fn)
print("TP", b_tp)
print("accuracy", f"{accuracy_score(y_test, baseline_pred):.3f}")
print("precision", f"{precision_score(y_test, baseline_pred, zero_division=0):.3f}")
print("recall", f"{recall_score(y_test, baseline_pred, zero_division=0):.3f}")

m_tn, m_fp, m_fn, m_tp = confusion_matrix(
    y_test, model.predict(X_test), labels=[0, 1]
).ravel()
baseline_value = value(b_tn, b_fp, b_fn, b_tp)
model_value = value(m_tn, m_fp, m_fn, m_tp)
diff_value = model_value - baseline_value
print("baseline total value", baseline_value)
print("baseline value per appointment", baseline_value / 120)
print("model total value", model_value)
print("model value per appointment", model_value / 120)
print("difference total", diff_value)
print("difference per appointment", diff_value / 120)


def should_act(appointment):
    row = pd.DataFrame([appointment], columns=X_train.columns)
    pred = int(model.predict(row)[0])
    action = "overbook" if pred == 1 else "hold"
    leaf_id = int(model.apply(row)[0])
    p_no_show = float(model.predict_proba(row)[0, 1])

    node_indicator = model.decision_path(row)
    node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    print("path")
    for node_id in node_index:
        if leaf_id == node_id:
            continue
        feat_name = X_train.columns[feature[node_id]]
        value = row.iloc[0, feature[node_id]]
        sign = "<=" if value <= threshold[node_id] else ">"
        print(f"node {node_id}: {feat_name} = {value} {sign} {threshold[node_id]}")

    return action, leaf_id, p_no_show


D = {
    "lead_time_days": 21,
    "prior_no_shows": 1,
    "age": 27,
    "distance_miles": 9.5,
    "sms_reminder": 0,
    "new_patient": 1,
    "appointment_hour": 15,
}
E = {
    "lead_time_days": 3,
    "prior_no_shows": 0,
    "age": 61,
    "distance_miles": 2.0,
    "sms_reminder": 1,
    "new_patient": 0,
    "appointment_hour": 9,
}

print("D")
action, leaf_id, p_no_show = should_act(D)
print("action", action)
print("leaf id", leaf_id)
print("probability of no-show", p_no_show)

print("E")
action, leaf_id, p_no_show = should_act(E)
print("action", action)
print("leaf id", leaf_id)
print("probability of no-show", p_no_show)
