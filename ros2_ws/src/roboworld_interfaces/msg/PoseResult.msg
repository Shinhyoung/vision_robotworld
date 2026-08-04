# 6D pose estimation result for a single captured frame.
# Published by: roboworld_pose/pose_node
# Contract owner: Team Lead. Do NOT change without an ICD revision.

std_msgs/Header header          # stamp = capture time of the source frame
string part_id
uint32 sequence

bool valid                      # false => pose fields are meaningless, check `message`
geometry_msgs/PoseStamped pose  # position in METERS, orientation as unit quaternion (x,y,z,w)
                                # pose.header.frame_id is authoritative (see ICD section 4)

float32 fitness                 # registration inlier ratio, [0.0, 1.0]; higher is better
float32 rmse_m                  # registration RMSE in meters; lower is better
float64[36] covariance          # row-major 6x6 [x y z rx ry rz], all-zero = not estimated

float32 inference_time_ms
string backend                  # "foundationpose" | "icp" | "stub"
string message                  # human readable diagnostic, "" when valid
