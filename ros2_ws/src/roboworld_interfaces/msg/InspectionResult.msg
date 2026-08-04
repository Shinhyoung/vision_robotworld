# Surface defect inspection result for a single captured frame.
# Published by: roboworld_inspection/inspection_node
# Contract owner: Team Lead. Do NOT change without an ICD revision.

std_msgs/Header header          # stamp = capture time of the source color frame
                                # frame_id = optical frame the image was taken in
string part_id                  # part type key, e.g. "guide_block" (see parts.yaml)
uint32 sequence                 # trigger sequence number, monotonically increasing

bool is_good                    # true = OK (pass), false = NG (defect)
float32 anomaly_score           # normalized anomaly score, [0.0, 1.0]
float32 threshold               # threshold that produced is_good, [0.0, 1.0]

sensor_msgs/Image anomaly_map   # 32FC1 heat map, empty (height==0) if publishing disabled
sensor_msgs/Image defect_mask   # mono8, 0 or 255, empty (height==0) if publishing disabled

float32 inference_time_ms       # backend inference wall time
string backend                  # "efficientad" | "statistical" | "stub"
