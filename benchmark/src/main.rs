use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

const MIDI_DIR: &str = "../test-asset";

const MIDI_EXT: &[&str] = &["mid", "midi", "rmi"];

fn parse_midly(data: &[u8]) -> Result<usize, String> {
    let smf = midly::Smf::parse(data).map_err(|err| format!("{}", err))?;
    Ok(smf.tracks.len())
}

fn list_midis(dir: &Path) -> Vec<PathBuf> {
    let mut midis = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if MIDI_EXT
            .iter()
            .any(|ext| path.extension() == Some(ext.as_ref()))
        {
            midis.push(path);
        }
    }
    midis
}

fn use_parser(parse: fn(&[u8]) -> Result<usize, String>, data: &[u8]) -> Result<(), String> {
    let round = |num: f64| (num * 100.0).round() / 100.0;

    let runtime = || -> Result<_, String> {
        let start = Instant::now();
        let out = parse(data)?;
        let time = round((start.elapsed().as_micros() as f64) / 1000.0);
        Ok((out, time))
    };

    let (track_count, cold_time) = runtime()?;
    let runtime = || -> Result<_, String> {
        let (out, time) = runtime()?;
        assert_eq!(
            out, track_count,
            "parser is not consistent with track counts"
        );
        Ok(time)
    };

    let iters = (2000.0 / cold_time).floor() as u64 + 1;
    let mut total_time = 0.0;
    let mut max_time = cold_time;
    let mut min_time = cold_time;
    for _ in 0..iters {
        let time = runtime()?;
        total_time += time;
        max_time = max_time.max(time);
        min_time = min_time.min(time);
    }
    let avg_time = round(total_time / (iters as f64));

    eprintln!(
        "{} tracks in {} iters / min {} / avg {} / max {}",
        track_count, iters, min_time, avg_time, max_time
    );

    Ok(())
}

fn main() {
    let midi_filter = env::args().nth(1).unwrap_or_default().to_lowercase();
    let midi_dir = env::args().nth(2).unwrap_or(MIDI_DIR.to_string());

    let unfiltered_midis = list_midis(midi_dir.as_ref());
    let midis = unfiltered_midis
        .iter()
        .filter(|midi| {
            midi.file_name()
                .unwrap_or_default()
                .to_str()
                .expect("non-utf8 file")
                .to_lowercase()
                .contains(&midi_filter)
        })
        .collect::<Vec<_>>();
    if midis.is_empty() {
        eprintln!("no midi files match the pattern \"{}\"", midi_filter);
        eprintln!("available midi files:");
        for file in unfiltered_midis.iter() {
            eprintln!("  {}", file.display());
        }
    } else {
        for midi in midis {
            //Read file once and reuse for all iterations
            let data = fs::read(midi).map_err(|err| format!("{}", err)).unwrap();
            let size = data.len();
            eprintln!("parsing file \"{}\" ({} KB)", midi.display(), size / 1024);
            eprint!("  midly: ");
            match use_parser(parse_midly, &data) {
                Ok(()) => {}
                Err(_err) => {
                    eprintln!("parse error");
                }
            }
            eprintln!();
        }
    }
}
