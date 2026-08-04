# FINAL OUTPUT MESSAGE consumed by the robot department.
# Published on: /roboworld/part_result   (see docs/robot_interface_ICD.md)
# Published by: roboworld_pipeline/pipeline_node
#
# THIS IS THE INTERFACE CONTRACT. Any field change requires an ICD revision,
# a version bump of `pipeline_version`, and notification of the robot department.

# --- status enumeration -------------------------------------------------
uint8 STATUS_OK=0         # inspected good AND pose valid  -> robot may pick
uint8 STATUS_NG=1         # defect detected, pose intentionally skipped -> reject part
uint8 STATUS_NO_POSE=2    # good part but pose estimation failed -> do not pick, re-trigger
uint8 STATUS_ERROR=3      # pipeline/hardware error -> do not pick, see `message`

std_msgs/Header header    # stamp = capture time of the frame this result derives from
                          # frame_id = camera optical frame the data originated in
uint32 sequence           # trigger sequence number, monotonically increasing
string part_id            # part type key, e.g. "guide_block"

uint8 status              # one of STATUS_*
bool is_good              # true only when status == STATUS_OK or the part passed inspection
float32 anomaly_score     # [0.0, 1.0]
float32 anomaly_threshold # [0.0, 1.0]

bool pose_valid           # true only when a usable 6D pose is present
geometry_msgs/PoseStamped pose  # meters + unit quaternion; frame_id authoritative
float32 pose_fitness      # [0.0, 1.0]

float32 tact_time_ms      # trigger -> publish wall time
string pipeline_version   # semantic version of the producing pipeline, e.g. "0.1.0"
string message            # human readable diagnostic, "" on nominal STATUS_OK
