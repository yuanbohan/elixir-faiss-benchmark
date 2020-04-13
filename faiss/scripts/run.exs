# {xb, xq} = Faiss.load_all_dataset()
# target_video_id = List.first(xq)
# cluster = Faiss.index(xb)
# result_video_id = Faiss.search(cluster, target_video_id)

# IO.puts(map_size(cluster))
# IO.puts(inspect(target_video_id))
# IO.puts(inspect(result_video_id))

{xb, xq} = Faiss.load_all_dataset()
target_video_id = List.first(xq)

IO.puts("start: Elixir index #{length(xb)} dataset")
{times, cluster} = :timer.tc(Faiss, :index, [xb])
IO.puts("end: Elixir index consumes #{times / 1_000_000} (s)")

IO.puts("start: Elixir search")
{times, _} = :timer.tc(Faiss, :search, [cluster, target_video_id])
IO.puts("end: Elixir search consumes #{times / 1_000_000} (s)")
