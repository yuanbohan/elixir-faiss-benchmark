{xb, xq} = Faiss.load_all_dataset()

target_video_id = List.first(xq)

cluster = Faiss.index(xb)
result_video_id = Faiss.search(cluster, target_video_id)

# IO.puts(map_size(cluster))
# IO.puts(inspect(target_video_id))
# IO.puts(inspect(result_video_id))
