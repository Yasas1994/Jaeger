from preprocess.shuffle_dna import dinuc_shuffle
import polars as pl
import pyfastx 

def shuffle_core(**kwargs):
    

    match kwargs.get('itype'):
        case 'CSV':
            f = pl.read_csv(kwargs.get('input'),
                            truncate_ragged_lines=True,
                            has_header=False)
            f = f.with_columns(
                pl.lit(1).alias("column_1")
            )
            fs = f.with_columns(
                pl.col("column_2")
            .map_elements(lambda x : dinuc_shuffle(x) , return_dtype=pl.String),
                pl.lit(0).alias("column_1")
            )
            f = pl.concat([f, fs]).sample(fraction=1.0, shuffle=True, with_replacement=False)
            f.write_csv(kwargs.get('output'), include_header=False)
        case 'FASTA':
            pass
            f = pyfastx.Fasta(kwargs.get('input'), build_index=False)
