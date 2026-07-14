-- Meeting-level speaker display-name aliases (JSON map).
-- Example: {"speaker_1":"Alice","you":"Mendel"}
-- Segment speaker IDs in transcripts remain unchanged.

ALTER TABLE meetings ADD COLUMN speaker_aliases TEXT;
