defmodule Faiss do
  @moduledoc """
  help to load data to memory, cluster all the data, search k nearest neighbors
  """

  @distance_threshold 5

  @type distance :: non_neg_integer()
  @type video_id() :: binary()
  @type cluster() :: [video_id()]
  @type store() :: %{optional(video_id()) => [video_id()]}

  @doc """
  convert comma seperated uint8 list to binary

  ## Examples

      iex> Faiss.line_to_binary("1,2,3")
      <<1, 2, 3>>

  """
  @spec line_to_binary(String.t()) :: video_id()
  def line_to_binary(line) do
    l = line |> String.split(",") |> Enum.map(&String.to_integer/1)
    for x <- l, into: "", do: <<x>>
  end

  @doc """
  load dataset from disk and convert line to binary.

  NOTE: dataset is generated by python script usind `NumPy` lib
  """
  @spec load_one_dataset(String.t()) :: [video_id()]
  def load_one_dataset(path) do
    {:ok, contents} = File.read(path)
    contents |> String.split("\n", trim: true) |> Enum.map(&line_to_binary/1)
  end

  @spec load_all_dataset :: {[video_id()], [video_id()]}
  def load_all_dataset do
    # xb = load_one_dataset("../xb.txt")
    xq = load_one_dataset("../xq.txt")

    {[<<1>>], xq}
  end

  @doc """
  cal hamming distance between 2 binaries

  ## Examples

      iex> Faiss.hamming_distance(<<1>>, <<7>>)
      2

  """
  @spec hamming_distance(video_id(), video_id()) :: distance()
  def hamming_distance(video_id1, video_id2) do
    bits = :crypto.exor(video_id1, video_id2)
    for(<<bit::1 <- bits>>, do: bit) |> Enum.sum()
  end

  @doc """
  `cluster` store the group id, `map` store the clustered id

  # FIXME: the `Enum.reverse` may have a performance issue

  ## Examples

      iex> Faiss.add({[], %{}}, <<1>>)
      {[<<1>>], %{<<1>> => []}}

      iex> Faiss.add({[<<1>>], %{<<1>> => []}}, <<2>>)
      {[<<1>>], %{<<1>> => [<<2>>]}}

  """
  @spec add({cluster(), store()}, video_id()) :: {cluster(), store()}
  def add({cluster, store}, video_id) do
    exist_group_id =
      cluster
      |> Enum.reverse()
      |> Enum.find(fn group_id -> hamming_distance(group_id, video_id) < @distance_threshold end)

    case exist_group_id do
      nil -> {[video_id | cluster], Map.put(store, video_id, [])}
      _ -> {cluster, Map.update(store, exist_group_id, [], &[video_id | &1])}
    end
  end

  @doc """
  cluster the dataset based on hammind distance

  NOTE: the result cluster MUST be reversed before search.
  """
  @spec index([video_id()]) :: {cluster(), store()}
  def index(dataset) do
    Enum.reduce(dataset, {[], %{}}, fn video_id, acc -> add(acc, video_id) end)
  end

  @doc """
  find the video_id which has shortest hamming distance with the `video_id`

  Assume:

  - the `cluster` is already reversed
  - the `cluster` is not empty

  """
  @spec search({cluster(), store()}, video_id()) :: video_id()
  def search({cluster, store}, target_id) do
    group_id =
      Enum.reduce(cluster, fn group_id, acc ->
        if hamming_distance(group_id, target_id) < hamming_distance(acc, target_id) do
          group_id
        else
          acc
        end
      end)

    group_videos = [group_id | Map.get(store, group_id, [])]
    Enum.min_by(group_videos, fn video_id -> hamming_distance(video_id, target_id) end)
  end
end