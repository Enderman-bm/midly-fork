//! Internal fast MIDI parsing utilities used by the note loader.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    error::Result,
    primitive::{Format, Fps, Timing, u15},
    ErrorKind,
};

use core::convert::TryInto;

#[cfg(feature = "alloc")]
use crate::riff;

/// A lightweight MIDI event used by the fast parser.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MidiEvent<'a> {
    NoteOn { _channel: u8, key: u8, velocity: u8 },
    NoteOff { _channel: u8, key: u8, _velocity: u8 },
    Meta { event_type: u8, data: &'a [u8] },
    Other,
}

/// A lightweight track iterator over raw MIDI track bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrackIter<'a> {
    data: &'a [u8],
    offset: usize,
    last_status: Option<u8>,
}

impl<'a> TrackIter<'a> {
    #[inline]
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            offset: 0,
            last_status: None,
        }
    }

    pub(crate) fn next_event(&mut self) -> Option<(u32, MidiEvent<'a>)> {
        if self.offset >= self.data.len() {
            return None;
        }

        let delta_time = self.read_vlq();
        if self.offset >= self.data.len() {
            return None;
        }

        let mut status = self.data[self.offset];
        if status < 0x80 {
            status = self.last_status?;
        } else {
            self.offset += 1;
            if status < 0xF0 {
                self.last_status = Some(status);
            }
        }

        let event = match status & 0xF0 {
            0x80 => {
                let key = self.data.get(self.offset).copied().unwrap_or(0);
                let velocity = self.data.get(self.offset + 1).copied().unwrap_or(0);
                self.offset = self.offset.saturating_add(2);
                MidiEvent::NoteOff {
                    _channel: status & 0x0F,
                    key,
                    _velocity: velocity,
                }
            }
            0x90 => {
                let key = self.data.get(self.offset).copied().unwrap_or(0);
                let velocity = self.data.get(self.offset + 1).copied().unwrap_or(0);
                self.offset = self.offset.saturating_add(2);
                if velocity == 0 {
                    MidiEvent::NoteOff {
                        _channel: status & 0x0F,
                        key,
                        _velocity: 0,
                    }
                } else {
                    MidiEvent::NoteOn {
                        _channel: status & 0x0F,
                        key,
                        velocity,
                    }
                }
            }
            0xF0 => {
                if status == 0xFF {
                    let meta_type = self.data.get(self.offset).copied().unwrap_or(0);
                    self.offset = self.offset.saturating_add(1);
                    let len = self.read_vlq() as usize;
                    let end = (self.offset + len).min(self.data.len());
                    let data = &self.data[self.offset..end];
                    self.offset = end;
                    MidiEvent::Meta {
                        event_type: meta_type,
                        data,
                    }
                } else if status == 0xF0 || status == 0xF7 {
                    let len = self.read_vlq() as usize;
                    let end = (self.offset + len).min(self.data.len());
                    self.offset = end;
                    MidiEvent::Other
                } else {
                    MidiEvent::Other
                }
            }
            _ => {
                // Skip the data bytes for channel messages we don't care about.
                match status & 0xF0 {
                    0xA0 | 0xB0 | 0xE0 => {
                        self.offset = self.offset.saturating_add(2);
                    }
                    0xC0 | 0xD0 => {
                        self.offset = self.offset.saturating_add(1);
                    }
                    _ => {}
                }
                MidiEvent::Other
            }
        };

        Some((delta_time, event))
    }

    fn read_vlq(&mut self) -> u32 {
        let mut value = 0u32;
        loop {
            if self.offset >= self.data.len() {
                break;
            }
            let byte = self.data[self.offset];
            self.offset = self.offset.saturating_add(1);
            value = (value << 7) | (byte & 0x7F) as u32;
            if (byte & 0x80) == 0 {
                break;
            }
        }
        value
    }
}

/// Parse the MIDI header and return (Header, track_count, division_raw, stripped_bytes).
#[cfg(feature = "alloc")]
pub(crate) fn parse_header<'a>(raw: &'a [u8]) -> Result<(crate::smf::Header, u16, u16, &'a [u8])> {
    let raw = match raw.get(..4) {
        Some(b"RIFF") => riff::unwrap(raw)?,
        Some(b"MThd") => raw,
        _ => return Err(crate::Error::new(err_invalid!("not a midi file"))),
    };

    if raw.len() < 14 {
        return Err(crate::Error::new(err_invalid!("invalid midi header")));
    }

    if &raw[0..4] != b"MThd" {
        return Err(crate::Error::new(err_invalid!("invalid midi header chunk")));
    }

    let _header_len = u32::from_be_bytes(raw[4..8].try_into().unwrap_or([0; 4])) as usize;
    let format_raw = u16::from_be_bytes(raw[8..10].try_into().unwrap_or([0; 2]));
    let tracks_count = u16::from_be_bytes(raw[10..12].try_into().unwrap_or([0; 2]));
    let division_raw = u16::from_be_bytes(raw[12..14].try_into().unwrap_or([0; 2]));

    let format = match format_raw {
        0 => Format::SingleTrack,
        1 => Format::Parallel,
        2 => Format::Sequential,
        _ => return Err(crate::Error::new(err_invalid!("invalid smf format"))),
    };

    let timing = if (division_raw & 0x8000) != 0 {
        let fps_raw = ((division_raw >> 8) & 0xFF) as u8;
        let fps = (-(fps_raw as i8)) as u8;
        let subframe = (division_raw & 0xFF) as u8;
        let fps = Fps::from_int(fps).ok_or_else(|| crate::Error::new(err_invalid!("invalid smpte fps")))?;
        Timing::Timecode(fps, subframe)
    } else {
        Timing::Metrical(u15::from(division_raw & 0x7FFF))
    };

    Ok((crate::smf::Header { format, timing }, tracks_count, division_raw, raw))
}

/// Iterate tracks from raw SMF data.
#[cfg(feature = "alloc")]
pub(crate) fn iter_tracks_from_data<'a>(data: &'a [u8], tracks_count: u16) -> Vec<TrackIter<'a>> {
    let mut tracks = Vec::new();
    let mut offset = 8 + 6;

    for _ in 0..tracks_count {
        if offset + 8 > data.len() {
            break;
        }
        while offset + 8 <= data.len() && &data[offset..offset + 4] != b"MTrk" {
            if offset + 8 > data.len() {
                break;
            }
            let len = u32::from_be_bytes(data[offset + 4..offset + 8].try_into().unwrap_or([0; 4])) as usize;
            offset = offset.saturating_add(8 + len);
        }
        if offset + 8 > data.len() {
            break;
        }

        let len = u32::from_be_bytes(data[offset + 4..offset + 8].try_into().unwrap_or([0; 4])) as usize;
        let end = (offset + 8 + len).min(data.len());
        let track_data = &data[offset + 8..end];
        tracks.push(TrackIter::new(track_data));
        offset = offset.saturating_add(8 + len);
    }

    tracks
}
