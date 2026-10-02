"""Constants shared across the pipeline. Conventions are documented in db/migrations/001_init.sql."""

# Strike zone (feet). Fixed zone: |side| <= 0.71, 1.5 <= height <= 3.0.
ZONE_HALF_WIDTH = 0.71
ZONE_BOTTOM = 1.5
ZONE_TOP = 3.0

SWING_CALLS = {"StrikeSwinging", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable", "FoulTip", "InPlay"}
WHIFF_CALLS = {"StrikeSwinging"}
CALLED_STRIKE_CALLS = {"StrikeCalled"}

# Batted-ball rows trusted for exit-velo metrics.
GOOD_LAUNCH_CONF = {"High", "Medium"}
# Pitch rows trusted for movement / location features.
BAD_PITCH_CONF = {"Low"}

PLAY_RESULT_MAP = {
    "Single": "1B",
    "Double": "2B",
    "Triple": "3B",
    "HomeRun": "HR",
    "Error": "ROE",
    "FieldersChoice": "FC",
    "Out": "OUT",
}

# Trackman ID shape (12- and 13-digit IDs both occur). Anything Excel touched (1.00E+11, 123.0) fails.
ID_PATTERN = r"^\d{6,15}$"

# A file is a Trackman pitch log if it has these. PitchUID/GameUID are generated when absent
# (some exports drop them); the date can come from Date, UTCDateTime/LocalDateTime or GameID.
REQUIRED_COLUMNS = [
    "Pitcher", "PitcherId", "PitcherThrows", "Batter", "BatterId", "BatterSide", "PitchCall", "Balls", "Strikes",
]

# Trackman column -> pitches column, for values copied as-is.
TEXT_COLUMNS = {
    "PitchUID": "pitch_uid",
    "GameUID": "game_uid",
    "Top/Bottom": "top_bottom",
    "Pitcher": "pitcher_name",
    "PitcherTeam": "pitcher_team",
    "Batter": "batter_name",
    "BatterTeam": "batter_team",
    "TaggedPitchType": "tagged_pitch_type",
    "PitchCall": "pitch_call",
    "KorBB": "kor_bb",
    "TaggedHitType": "tagged_hit_type",
    "PlayResult": "play_result",
    "HitLaunchConfidence": "hit_launch_conf",
}
INT_COLUMNS = {
    "PitchNo": "pitch_no",
    "Inning": "inning",
    "PAofInning": "pa_of_inning",
    "PitchofPA": "pitch_of_pa",
    "Outs": "outs",
    "Balls": "balls",
    "Strikes": "strikes",
    "OutsOnPlay": "outs_on_play",
    "RunsScored": "runs_scored",
}
FLOAT_COLUMNS = {
    "RelSpeed": "rel_speed",
    "SpinRate": "spin_rate",
    "SpinAxis": "spin_axis",
    "RelHeight": "rel_height",
    "RelSide": "rel_side",
    "Extension": "extension",
    "VertRelAngle": "vert_rel_angle",
    "HorzRelAngle": "horz_rel_angle",
    "InducedVertBreak": "induced_vert_break",
    "VertBreak": "vert_break",
    "HorzBreak": "horz_break",
    "PlateLocHeight": "plate_loc_height",
    "PlateLocSide": "plate_loc_side",
    "ZoneSpeed": "zone_speed",
    "VertApprAngle": "vert_appr_angle",
    "HorzApprAngle": "horz_appr_angle",
    "ZoneTime": "zone_time",
    "ExitSpeed": "exit_speed",
    "Angle": "launch_angle",
    "Direction": "direction",
    "Distance": "distance",
}

# Metrics that must be present for a pitch to count as tracked.
TRACKING_REQUIRED = ["rel_speed", "induced_vert_break", "horz_break", "plate_loc_side", "plate_loc_height",
                     "rel_height", "rel_side", "vert_appr_angle"]
