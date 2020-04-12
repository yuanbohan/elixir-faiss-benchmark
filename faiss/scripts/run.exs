{xb, xq} = Faiss.load_all_dataset()

target_video_id = List.first(xq)

{cluster, store} = Faiss.index(xq)
reversed_cluster = Enum.reverse(cluster)
result_video_id = Faiss.search({reversed_cluster, store}, target_video_id)

IO.puts(inspect(reversed_cluster))
IO.puts(inspect(store))
IO.puts(inspect(target_video_id))
IO.puts(inspect(result_video_id))
