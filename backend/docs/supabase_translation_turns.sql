-- SignSage translation_turns — minimal user-associated record foundation
-- ponytail: RLS enforces user isolation; no custom backend table logic needed

create table if not exists public.translation_turns (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  dir text not null check (dir in ('sign→text','text→pose','error')),
  text text not null,
  gloss text,
  latency_ms double precision,
  created_at timestamptz not null default now()
);

alter table public.translation_turns enable row level security;

drop policy if exists "Users can manage own turns" on public.translation_turns;
create policy "Users can manage own turns" on public.translation_turns
  for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index if not exists translation_turns_user_created_idx on public.translation_turns(user_id, created_at desc);
